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
        score, sources, metrics = _validate(model, dataset, config, device, occlusion, progress, step)
    if observer is not None:
        observer.last_dev_sources = sources
        observer.last_dev_input_stats = difference(runtime, input_before)
        if metrics.get("execution"):
            execution = metrics["execution"]
            observer.last_dev_input_stats.update(execution["host"])
            observer.last_dev_input_stats.update(execution["remote"])
            observer.last_dev_input_stats.update(case_host_frames=execution["host_frames"],
                                                 case_dgx_frames=execution["dgx_frames"])
        observer.last_dev_metrics = metrics
    if metrics is not None:
        message("DEV METRICS", {"step": step, **metrics["macro_mean"],
                                "by_dataset": metrics["by_dataset"]})
    return score


def _validate(model, dataset, config, device, occlusion, progress, step=None):
    from explore_a.evaluation.live import collect
    result, _ = collect(model, dataset, config, device, occlusion, predict_fn=predict, progress=progress, step=step)
    return result["selection_score"], result["selection_by_dataset"], result
