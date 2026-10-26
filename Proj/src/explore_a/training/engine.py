"""Train small heads while frozen models compute each sampled raw pair on demand."""
import copy
import gc
import math
from pathlib import Path
import time

import torch

from depth_project.io import write_json
from explore_a.observability.observer import RunObserver
from explore_a.observability.progress import Progress, message
from explore_a.online.factory import make_runtime
from explore_a.online.identity import input_identity, teacher_identity
from explore_a.online.loading import OnlinePairs
from explore_a.online.statistics import snapshot, difference
from explore_a.config import bind_identity, experiment_name, file_identity
from explore_a.data.prepare import load_manifest
from explore_a.geometry.warp import pair_geometry
from explore_a.models.occlusion import OcclusionStudent
from explore_a.training.checkpoint import load_checkpoint, save_checkpoint, seed_everything
from explore_a.training.losses import fusion_loss, student_loss
from explore_a.training.monitor import HeldoutMonitor, scheduled
from explore_a.training.prediction import load_student, make_fusion, predict
from explore_a.training.sampling import StepBatches
from explore_a.training.validation import validate


def schedule(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return .5 * (1 + math.cos(math.pi * min(progress, 1)))


def run_training(config, flow, occlusion="rule", fusion="unet", distillation=False, device="cuda"):
    retry_needed = False
    try:
        _run(config, flow, occlusion, fusion, distillation, device)
    except torch.cuda.OutOfMemoryError:
        if config["training"]["microbatch"] != 8:
            raise
        retry_needed = True
    # Leave the except block first, releasing the traceback and its large CUDA tensors.
    if retry_needed:
        gc.collect()
        torch.cuda.empty_cache()
        retry = copy.deepcopy(config)
        retry["training"]["microbatch"] = 4
        message("CUDA OOM: resume with microbatch 4, accumulation 2.")
        _run(retry, flow, occlusion, fusion, distillation, device)


def _run(config, flow, occlusion, fusion, distillation, device):
    settings = config["training"]
    seed_everything(settings["seed"])
    manifest = load_manifest(config)
    provenance = input_identity(config, flow)
    needs_teacher = distillation or occlusion == "mftiq"
    runtime = make_runtime(config, [flow], needs_teacher, device)
    if hasattr(runtime, "require_rows"):
        runtime.require_rows(manifest["samples"])
    # Random temporary initialization inside frozen libraries must not change head initialization.
    seed_everything(settings["seed"])
    datasets = {split: OnlinePairs(config, [r for r in manifest["samples"] if r["split"] == split],
                                  runtime, flow, needs_teacher) for split in ("train", "dev")}
    name = f"occlusion-{flow}" if distillation else experiment_name(flow, occlusion, fusion)
    root = Path(config["paths"]["runs"]) / name
    model = OcclusionStudent() if distillation else make_fusion(fusion, config)
    model.to(device).train()
    student = None
    if occlusion == "student" and not distillation:
        student = load_student(config, flow, device)
    # Observation frequency is not an optimization hyperparameter; online checkpoints can resume.
    training_identity = {k: v for k, v in settings.items()
                         if k not in {"microbatch", "workers", "heldout_every"}}
    identity = {"input": provenance, "training": training_identity,
                "experiment": {"flow": flow,
                               "occlusion": "distillation" if distillation else occlusion,
                               "fusion": "student" if distillation else fusion}}
    if needs_teacher:
        identity["teacher"] = teacher_identity(config)
    if student is not None:
        identity["student"] = file_identity(Path(config["paths"]["runs"])
                                            / f"occlusion-{flow}/best.pt")
    bind_identity(root / "identity.json", identity)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: schedule(
        step, settings["steps"], settings["warmup_steps"]))
    amp = device == "cuda" and settings["amp"]
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best = 0, float("inf")
    last = root / "last.pt"
    if last.exists():
        checkpoint = load_checkpoint(last, model, optimizer, scheduler, scaler, identity)
        start, best = checkpoint["step"], checkpoint["best"]
    else:
        save_checkpoint(last, model, optimizer, scheduler, scaler, 0, best, identity)
    with RunObserver(root, config, start) as observer:
        monitor = None
        if settings["heldout_every"] > 0:
            test = OnlinePairs(config, [r for r in manifest["samples"] if r["split"] == "test"],
                               runtime, flow, needs_teacher)
            monitor = HeldoutMonitor(root, test, config, identity, occlusion, student,
                                     distillation, device, observer=observer)
            monitor.maybe_run(model, start)
        if start >= settings["steps"]:
            message(f"Already complete: {name}")
            return
        _updates(config, datasets, model, student, optimizer, scheduler, scaler, identity,
                 root, start, best, amp, occlusion, distillation, device, monitor, observer)


def _updates(config, datasets, model, student, optimizer, scheduler, scaler, identity,
             root, start, best, amp, occlusion, distillation, device, monitor, observer):
    settings = config["training"]
    batches = StepBatches(datasets["train"].rows, settings["effective_batch"],
                          settings["microbatch"], settings["steps"], start, settings["seed"])
    iterator = datasets["train"].iter_batches(batches, ordered=False)
    accumulation = settings["effective_batch"] // settings["microbatch"]
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    try:
        with Progress(config, f"TRAIN {root.name}", settings["steps"], start, "step") as progress:
            _loop(config, datasets, model, student, optimizer, scheduler, scaler, identity,
                  root, start, best, amp, occlusion, distillation, device, monitor, observer,
                  iterator, accumulation, began, progress)
    finally:
        iterator.close()


def _loop(config, datasets, model, student, optimizer, scheduler, scaler, identity,
          root, start, best, amp, occlusion, distillation, device, monitor, observer,
          iterator, accumulation, began, progress):
    settings = config["training"]
    for step in range(start, settings["steps"]):
        runtime = datasets["train"].runtime
        input_before = snapshot(runtime)
        update_began = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accumulation):
            indices, data = next(iterator)
            with torch.autocast(device_type=device, enabled=amp):
                if distillation:
                    geometry = pair_geometry(data, config["geometry"])
                    loss = student_loss(model(data, geometry), data, geometry)
                else:
                    output, _ = predict(model, data, config, occlusion, student)
                    loss = fusion_loss(output, data, settings)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at update {step + 1}")
            scaler.scale(loss / accumulation).backward()
            total += float(loss.detach()) / accumulation
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        finished = step + 1
        update_s = time.monotonic() - update_began
        record = {"step": finished, "loss": total, "lr": scheduler.get_last_lr()[0],
                  "update_s": update_s, "pairs_per_s": settings["effective_batch"] / update_s,
                  "elapsed_s": time.monotonic() - began, "microbatch": settings["microbatch"]}
        if snapshot(runtime):
            record["input_stats"] = difference(runtime, input_before)
            values = record["input_stats"]
            progress.set(cache_hits=sum(v for k, v in values.items() if k.endswith("_hit")),
                         online=sum(v for k, v in values.items() if k.endswith("_online")))
        if device == "cuda":
            record["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        row = datasets["train"].rows[indices[-1]]
        progress.update(1, phase="train", loss=total, lr=record["lr"],
                        dataset=row["dataset"], sequence=row["sequence"],
                        pairs_s=record["pairs_per_s"],
                        **({"GPU_GiB": record["peak_allocated_gib"]} if device == "cuda" else {}))
        if finished % settings["validate_every"] == 0 or finished == settings["steps"]:
            progress.set(phase="dev")
            evaluated = time.monotonic()
            observer.last_dev_sources = {}
            score = validate(model, datasets["dev"], config, device, occlusion, student,
                             distillation, observer=observer, step=finished)
            record["dev_bce" if distillation else "dev_abs_rel"] = score
            if score < best:
                best = score
                save_checkpoint(root / "best.pt", model, optimizer, scheduler, scaler,
                                finished, best, identity)
            observer.development(finished, score, best, "bce" if distillation else "abs_rel",
                                 time.monotonic() - evaluated)
            progress.set(dev=score, best_dev=best)
        if (finished % settings["log_every"] == 0 or finished % settings["validate_every"] == 0
                or finished == settings["steps"]):
            observer.train(record)
        if finished % settings["save_every"] == 0 or finished == settings["steps"]:
            save_checkpoint(root / "last.pt", model, optimizer, scheduler, scaler,
                            finished, best, identity)
        if monitor is not None:
            if scheduled(finished, settings["steps"], settings["heldout_every"]):
                progress.set(phase="test")
            monitor.maybe_run(model, finished)
            if monitor.records:
                metrics = monitor.records[-1]["macro_mean"]
                progress.set(test=metrics.get("abs_rel", metrics.get("heldout_bce")))
        progress.set(phase="train")
        observer.plot(finished)
    write_json(root / "complete.json", {"steps": settings["steps"], "best_dev_score": best,
                                        "selection": "BCE" if distillation else "AbsRel"})
