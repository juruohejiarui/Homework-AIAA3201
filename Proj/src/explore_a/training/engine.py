"""Train small heads while frozen models compute each sampled raw pair on demand."""
from collections import Counter
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
from explore_a.online.window import InputWindow
from explore_a.training.two_frame import supervised
from explore_a.config import bind_identity, experiment_name
from explore_a.data.prepare import load_manifest
from explore_a.evaluation.depth import MetricSummary
from explore_a.training.checkpoint import load_checkpoint, save_checkpoint, seed_everything
from explore_a.training.dev_subset import select_dev
from explore_a.training.metrics import add_depth_scores, finish_depth_scores
from explore_a.training.correction_metrics import CorrectionMetrics
from explore_a.training.monitor import HeldoutMonitor
from explore_a.training.prediction import make_fusion
from explore_a.training.sampling import StepBatches
from explore_a.training.validation import validate
from explore_a.frontend import relative


def schedule(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return .5 * (1 + math.cos(math.pi * min(progress, 1)))


def run_training(config, flow, occlusion="rule", fusion="unet_guided", device="cuda"):
    retry_needed = False
    try:
        _run(config, flow, occlusion, fusion, device)
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
        _run(retry, flow, occlusion, fusion, device)


def _run(config, flow, occlusion, fusion, device):
    settings = config["training"]
    name = experiment_name(flow, occlusion, fusion)
    root = Path(config["paths"]["runs"]) / name
    from explore_a.cache.read_policy import enabled, settings as cache_read_settings
    from explore_a.remote_online.policy import settings as remote_settings
    message("TRAIN INPUT", {"config": config.get("config_path"),
                            "frontend": config.get("frontend", {"name": "da2"}),
                            "backend": config.get("execution", {}).get("backend", "online"),
                            "sampling_mode": settings.get("sampling_mode", "fix"),
                            "source_weights": settings.get("source_weights"),
                            "cache_read": cache_read_settings(config) if enabled(config) else None,
                            "remote_online": remote_settings(config),
                            "evaluation_backend": "same",
                            "results": str(root)})
    seed_everything(settings["seed"])
    manifest = load_manifest(config)
    provenance = input_identity(config, flow)
    needs_teacher = occlusion == "mftiq"
    runtime = make_runtime(config, [flow], needs_teacher, device)
    # Random temporary initialization inside frozen libraries must not change head initialization.
    seed_everything(settings["seed"])
    selected_rows = {split: [r for r in manifest["samples"] if r["split"] == split]
                     for split in ("train", "dev")}
    full_dev_count = len(selected_rows["dev"])
    selected_rows["dev"] = select_dev(selected_rows["dev"],
                                      settings.get("dev_sample_fraction", 1.),
                                      config["data"]["seed"])
    message("DEV INPUT", len(selected_rows["dev"]), "/", full_dev_count, "fixed pairs")
    datasets = {split: OnlinePairs(config, rows, runtime, flow, needs_teacher, phase=split)
                for split, rows in selected_rows.items()}
    model = make_fusion(fusion, config)
    model.to(device).train()
    # Observation frequency is not an optimization hyperparameter; online checkpoints can resume.
    training_identity = {k: v for k, v in settings.items()
                         if k not in {"microbatch", "workers", "heldout_every"}}
    identity = {"input": provenance, "training": training_identity,
                "experiment": {"flow": flow,
                               "occlusion": occlusion,
                               "fusion": fusion}}
    from explore_a.frontend import relative, metadata
    if relative(config):
        identity["frontend"] = metadata(config)
    if needs_teacher:
        identity["teacher"] = teacher_identity(config)
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
        old_fraction = checkpoint["identity"].get("training", {}).get("dev_sample_fraction", 1.)
        if old_fraction != settings.get("dev_sample_fraction", 1.):
            best = float("inf")
            message("DEV subset changed; best score restarts with the next dev evaluation.")
        # Keep optimizer moments while applying the current config's LR and decay.
        scheduler.base_lrs = [settings["learning_rate"] for _ in optimizer.param_groups]
        for group in optimizer.param_groups:
            group["initial_lr"] = settings["learning_rate"]
            group["lr"] = settings["learning_rate"] * schedule(
                start, settings["steps"], settings["warmup_steps"])
            group["weight_decay"] = settings["weight_decay"]
    else:
        save_checkpoint(last, model, optimizer, scheduler, scaler, 0, best, identity)
    with RunObserver(root, config, start) as observer:
        monitor = None
        if settings["heldout_every"] > 0:
            test = OnlinePairs(config, [r for r in manifest["samples"] if r["split"] == "test"],
                               runtime, flow, needs_teacher, phase="test")
            monitor = HeldoutMonitor(root, test, config, identity, occlusion,
                                     device=device, observer=observer)
            monitor.maybe_run(model, start)
        if start >= settings["steps"]:
            message(f"Already complete: {name}")
            runtime.close()
            return
        _updates(config, datasets, model, optimizer, scheduler, scaler, identity,
                 root, start, best, amp, occlusion, device, monitor, observer)


def _updates(config, datasets, model, optimizer, scheduler, scaler, identity,
             root, start, best, amp, occlusion, device, monitor, observer):
    settings = config["training"]
    batches = StepBatches(datasets["train"].rows, settings["effective_batch"],
                          settings["microbatch"], settings["steps"], start, settings["seed"],
                          settings.get("source_weights"), settings.get("sampling_mode", "fix"))
    iterator = datasets["train"].iter_batches(batches, ordered=False)
    accumulation = settings["effective_batch"] // settings["microbatch"]
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    try:
        with Progress(config, f"TRAIN {root.name}", settings["steps"], start, "step") as progress:
            _loop(config, datasets, model, optimizer, scheduler, scaler, identity,
                  root, start, best, amp, occlusion, device, monitor, observer,
                  iterator, accumulation, began, progress)
    finally:
        try:
            iterator.close()
        finally:
            datasets["train"].runtime.close()


def _loop(config, datasets, model, optimizer, scheduler, scaler, identity,
          root, start, best, amp, occlusion, device, monitor, observer,
          iterator, accumulation, began, progress):
    settings = config["training"]
    window = InputWindow(datasets["train"].runtime)
    for step in range(start, settings["steps"]):
        runtime = datasets["train"].runtime
        input_before = snapshot(runtime)
        update_began = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        loss_values = Counter()
        depth_scores = MetricSummary()
        corrections = CorrectionMetrics(settings["correction_fraction"])
        source_losses = {}
        for _ in range(accumulation):
            indices, data = next(iterator)
            with torch.autocast(device_type=device, enabled=amp):
                output, terms = supervised(model, data, config, occlusion, step + 1)
                loss = terms.total
            term_values = terms.scalars()
            for key, value in term_values.items():
                loss_values[key] += value / accumulation
            batch_rows = [datasets["train"].rows[index] for index in indices]
            corrections.add(output, batch_rows)
            source = batch_rows[0]["dataset"]
            values = source_losses.setdefault(source, Counter())
            for key, value in term_values.items():
                values[key] += value / accumulation
            add_depth_scores(depth_scores, output, data,
                             [datasets["train"].rows[index] for index in indices], config)
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
        record.update(loss_values)
        if terms.temporal is not None:
            warmup = max(settings.get("temporal_warmup_steps", 500), 1)
            record["temporal_coefficient"] = settings.get("temporal_weight", .05) * min(
                1., finished / warmup)
        record["correction_metrics"] = corrections.finish()
        record["losses_by_dataset"] = source_losses
        record["depth_metrics"] = finish_depth_scores(depth_scores)
        if snapshot(runtime):
            record["input_stats"] = difference(runtime, input_before)
        if device == "cuda":
            record["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        window.capture(runtime, update_s)
        # tqdm supplies speed and ETA; keep its postfix limited to optimization scalars.
        progress.update(1, loss=total, lr=record["lr"])
        if finished % settings["validate_every"] == 0 or finished == settings["steps"]:
            evaluated = time.monotonic()
            observer.last_dev_sources = {}
            observer.last_dev_metrics = None
            score = validate(model, datasets["dev"], config, device, occlusion,
                             observer=observer, step=finished)
            from explore_a.frontend import relative
            metric_name = "ssi_abs_rel" if relative(config) else "abs_rel"
            record["dev_" + metric_name] = score
            if score < best:
                best = score
                save_checkpoint(root / "best.pt", model, optimizer, scheduler, scaler,
                                finished, best, identity)
            observer.development(finished, score, best, metric_name,
                                 time.monotonic() - evaluated)
        if (finished % settings["log_every"] == 0 or finished % settings["validate_every"] == 0
                or finished == settings["steps"]):
            record["input_window_stats"] = window.report()
            observer.train(record)
            window = InputWindow(runtime)
        if finished % settings["save_every"] == 0 or finished == settings["steps"]:
            save_checkpoint(root / "last.pt", model, optimizer, scheduler, scaler,
                            finished, best, identity)
        if monitor is not None:
            monitor.maybe_run(model, finished)
        observer.plot(finished)
    selection = "SSI AbsRel" if relative(config) else "AbsRel"
    write_json(root / "complete.json", {"steps": settings["steps"], "best_dev_score": best,
                                        "selection": selection})
