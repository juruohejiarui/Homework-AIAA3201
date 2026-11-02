"""Full held-out test scoring in memory, separate from checkpoint selection and final plots."""
import torch

from explore_a.data.reading import depth_range
from explore_a.evaluation.depth import MetricSummary, frame_scores
from explore_a.evaluation.state import evaluation_state
from explore_a.observability.progress import Progress
from explore_a.training.prediction import predict
from explore_a.training.correction_metrics import CorrectionMetrics


def collect_heldout(model, dataset, config, device, occlusion, include_baseline=False, step=None):
    """Evaluate only test rows; baseline and model share the exact same measured GT mask."""
    if not dataset.rows or any(row["split"] != "test" for row in dataset.rows):
        raise ValueError("Held-out monitor requires a nonempty dataset containing only test rows")
    scores = MetricSummary()
    baseline = MetricSummary() if include_baseline else None
    corrections = CorrectionMetrics(config["training"]["correction_fraction"])
    label = "TEST" if step is None else f"TEST step {step}"
    with Progress(config, label, len(dataset.rows)) as progress, evaluation_state(model):
        for indices, data in dataset.iter_batches():
            with torch.autocast(device_type=device, enabled=device == "cuda"
                                and config["training"]["amp"]):
                output, _ = predict(model, data, config, occlusion)
            corrections.add(output, [dataset.rows[index] for index in indices])
            for local, index in enumerate(indices):
                row = dataset.rows[index]
                mask = data["gt_valid"][local].bool()
                limits = depth_range(row["dataset"], config["data"])
                from explore_a.evaluation.depth import field_scores
                metrics = field_scores(output["depth"][local], data["gt"][local],
                                       mask, config, row["dataset"])
                scores.add(row, metrics, mask.sum())
                if baseline is not None:
                    raw = field_scores(data["current_depth"][local], data["gt"][local],
                                       mask, config, row["dataset"])
                    baseline.add(row, raw, mask.sum())
            row = dataset.rows[indices[-1]]
            progress.update(len(indices), dataset=row["dataset"], sequence=row["sequence"])
    result = scores.finish()
    result["correction_metrics"] = corrections.finish()
    return result, baseline.finish() if baseline is not None else None
