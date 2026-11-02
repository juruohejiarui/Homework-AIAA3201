"""Development-only checkpoint selection, macro averaged by sequence then dataset."""
from collections import defaultdict

import numpy as np
import torch

from explore_a.data.reading import depth_range
from explore_a.evaluation.depth import MetricSummary, frame_scores
from explore_a.evaluation.state import evaluation_state
from explore_a.observability.progress import Progress, message
from explore_a.training.prediction import predict
from explore_a.online.statistics import snapshot, difference
from explore_a.training.correction_metrics import CorrectionMetrics


def validate(model, dataset, config, device, occlusion,
             observer=None, step=None):
    label = "DEV" if step is None else f"DEV step {step}"
    runtime = getattr(dataset, "runtime", None)
    input_before = snapshot(runtime)
    with Progress(config, label, len(dataset.rows)) as progress, evaluation_state(model):
        score, sources, metrics = _validate(model, dataset, config, device, occlusion, progress)
    if observer is not None:
        observer.last_dev_sources = sources
        observer.last_dev_input_stats = difference(runtime, input_before)
        observer.last_dev_metrics = metrics
    if metrics is not None:
        message("DEV METRICS", {"step": step, **metrics["macro_mean"],
                                "by_dataset": metrics["by_dataset"]})
    return score


def _validate(model, dataset, config, device, occlusion, progress):
    scores = defaultdict(list)
    benchmark = MetricSummary()
    corrections = CorrectionMetrics(config["training"]["correction_fraction"])
    for indices, data in dataset.iter_batches():
        with torch.autocast(device_type=device, enabled=device == "cuda"
                            and config["training"]["amp"]):
            output, _ = predict(model, data, config, occlusion)
            corrections.add(output, [dataset.rows[index] for index in indices])
            for local, index in enumerate(indices):
                row = dataset.rows[index]
                valid = data["gt_valid"][local].bool()
                from explore_a.evaluation.depth import field_scores
                from explore_a.frontend import relative
                measured = field_scores(output["depth"][local], data["gt"][local], valid,
                                        config, row["dataset"])
                benchmark.add(row, measured, valid.sum())
                if valid.any():
                    metrics = ({"abs_rel": measured["ssi_abs_rel"]} if relative(config) else
                               frame_scores(output["depth"][local], data["gt"][local],
                                            valid, all_metrics=False))
                    row = dataset.rows[index]
                    scores[(row["dataset"], row["sequence"])].append(metrics["abs_rel"])
        row = dataset.rows[indices[-1]]
        progress.update(len(indices), dataset=row["dataset"], sequence=row["sequence"])
    by_source = defaultdict(list)
    for (source, _), values in scores.items():
        by_source[source].append(float(np.mean(values)))
    if not by_source:
        raise ValueError("No valid development supervision")
    score = float(np.mean([np.mean(values) for values in by_source.values()]))
    metrics = benchmark.finish()
    metrics["correction_metrics"] = corrections.finish()
    by_dataset = {source: float(np.mean(values)) for source, values in by_source.items()}
    return score, by_dataset, metrics
