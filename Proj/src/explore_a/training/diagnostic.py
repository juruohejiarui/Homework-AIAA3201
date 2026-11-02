"""Isolated 30-update cached-head learning check and GradNorm feasibility measurements."""
from pathlib import Path
import statistics
import time

import torch

from depth_project.io import write_json
from explore_a.training.cached_probe import load_probe, unchanged
from explore_a.training.checkpoint import seed_everything
from explore_a.training.correction_metrics import CorrectionMetrics
from explore_a.training.gradnorm import measure
from explore_a.training.losses import fusion_loss
from explore_a.training.prediction import make_fusion, predict
from explore_a.training.sampling import collate, to_device


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def comparison(model, batches, config, device):
    """Same fixed eight inputs, no optimizer update; two warmups and five timed repeats."""
    report = {}
    for mode in ("ordinary_backward", "gradient_measurement"):
        elapsed, peaks = [], []
        for repeat in range(7):
            model.zero_grad(set_to_none=True)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            synchronize(device)
            began = time.perf_counter()
            if mode == "gradient_measurement":
                measure(model, batches, config)
            else:
                for source, data in batches.items():
                    output, _ = predict(model, data, config, "rule")
                    loss = fusion_loss(output, data, config["training"]).total
                    (loss * (len(data["gt"]) / 8)).backward()
            synchronize(device)
            if repeat >= 2:
                elapsed.append(time.perf_counter() - began)
                if device == "cuda":
                    peaks.append(torch.cuda.max_memory_allocated() / 2**30)
        report[mode] = {"median_s": statistics.median(elapsed),
                        "peak_allocated_gib": max(peaks) if peaks else None}
    model.zero_grad(set_to_none=True)
    report["note"] = ("FP32 head forward/backward only; no optimizer update or frozen inference; "
                      "diagnostic includes two objective gradients, CPU export and scalar logging")
    return report


def run(config, output, device="cuda", steps=30):
    output = Path(output)
    if not 1 <= steps <= 30:
        raise ValueError("Diagnostic steps must be between 1 and 30")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty diagnostic output; formal checkpoints are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(config["training"]["seed"])
    pairs, skipped, inventory = load_probe(config)
    # Frozen outputs are loaded once into CPU memory and transferred once, with no online path.
    fixed = {source: to_device(collate([data for _, data in values]), device)
             for source, values in pairs.items()}
    model = make_fusion("flow_refine", config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                 weight_decay=config["training"]["weight_decay"])
    traces, probes, initial, timings = [], [], None, []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        began = time.perf_counter()
        losses, corrections = {}, CorrectionMetrics(config["training"]["correction_fraction"])
        for source, count in (("kitti", 4), ("tum", 2), ("nyuv2", 2)):
            size = len(pairs[source])
            indices = [(step * count + offset) % size for offset in range(count)]
            batch = {key: value[indices] for key, value in fixed[source].items()}
            prediction, _ = predict(model, batch, config, "rule")
            terms = fusion_loss(prediction, batch, config["training"])
            (terms.total * (count / 8)).backward()
            losses[source] = {"total": float(terms.total.detach()), **terms.scalars()}
            corrections.add(prediction, [pairs[source][i][0] for i in indices])
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
        optimizer.step()
        synchronize(device)
        elapsed = time.perf_counter() - began
        timings.append(elapsed)
        traces.append({"step": step, "losses": losses, "head_update_s": elapsed,
                       "correction_metrics": corrections.finish()})
        if step in (5, 10, 20, 30):
            # Match the measured training budget exactly: four/two/two pairs, not 18 pairs.
            # Repeat deterministically if a source has fewer usable pairs than its budget.
            batches = {}
            for source, count in (("kitti", 4), ("tum", 2), ("nyuv2", 2)):
                ids = [i % len(pairs[source]) for i in range(count)]
                batches[source] = {k: v[ids] for k, v in fixed[source].items()}
            synchronize(device)
            began = time.perf_counter()
            report = measure(model, batches, config, initial)
            synchronize(device)
            report.update(step=step, gradient_probe_s=time.perf_counter() - began)
            if initial is None:
                initial = report["losses"]
            probes.append(report)
            print("GradNorm probe", step, report["objectives"]["total"]["encoder"], flush=True)
    intact = unchanged(inventory)
    if not intact:
        raise ValueError("An inspected prediction NPZ changed during diagnostic")
    warm = timings[5:] or timings
    result = {"protocol": "FP32 cached head only; 4/2/2 pairs; no frozen inference or AMP",
              "sampling_note": "Fixed training-only pilot; not a generalization experiment",
              "alpha": 1.5, "weight_sum": 3, "weights_applied": False,
              "weight_note": "Instantaneous norm targets, not a trained GradNorm optimizer",
              "objective_note": "Unweighted source-mean gradients; weights are not sampling "
                                "probabilities or validated replacements for the fixed prior",
              "loss_rate_reference_step": 5, "steps": steps, "skipped": skipped,
              "samples": {s: [r["id"] for r, _ in v] for s, v in pairs.items()},
              "trace": traces, "probes": probes, "inspected_npz_unchanged": intact,
              "head_update_median_s": statistics.median(warm),
              "gradient_probe_median_s": statistics.median(
                  [p["gradient_probe_s"] for p in probes]) if probes else None,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30
              if device == "cuda" else None}
    write_json(output / "diagnostic.json", result)
    if probes:
        write_json(output / "comparison.json", comparison(model, batches, config, device))
    torch.save({"model": model.state_dict(), "step": steps,
                "purpose": "temporary diagnostic, not a formal run checkpoint"}, output / "head.pt")
    return result
