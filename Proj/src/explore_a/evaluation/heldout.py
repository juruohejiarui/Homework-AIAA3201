"""Full held-out test scoring in memory, separate from checkpoint selection and final plots."""
import torch

from explore_a.data.reading import depth_range
from explore_a.evaluation.depth import MetricSummary, frame_scores
from explore_a.evaluation.state import evaluation_state
from explore_a.geometry.warp import pair_geometry
from explore_a.observability.progress import Progress
from explore_a.training.losses import student_loss
from explore_a.training.prediction import predict


def collect_heldout(model, dataset, config, device, occlusion, student=None,
                    distillation=False, include_baseline=False, step=None):
    """Evaluate only test rows; baseline and model share the exact same measured GT mask."""
    if not dataset.rows or any(row["split"] != "test" for row in dataset.rows):
        raise ValueError("Held-out monitor requires a nonempty dataset containing only test rows")
    scores = MetricSummary()
    baseline = MetricSummary() if include_baseline and not distillation else None
    label = "TEST" if step is None else f"TEST step {step}"
    with Progress(config, label, len(dataset.rows)) as progress, evaluation_state(model, student):
        for indices, data in dataset.iter_batches():
            with torch.autocast(device_type=device, enabled=device == "cuda"
                                and config["training"]["amp"]):
                if distillation:
                    geometry = pair_geometry(data, config["geometry"])
                    logits = model(data, geometry)
                else:
                    output, _ = predict(model, data, config, occlusion, student)
            for local, index in enumerate(indices):
                row = dataset.rows[index]
                if distillation:
                    mask = geometry["hard"][local]
                    metrics = None
                    if mask.any():
                        part = {k: v[local:local + 1] for k, v in data.items()}
                        geo = {k: v[local:local + 1] for k, v in geometry.items()}
                        metrics = {"heldout_bce": float(student_loss(
                            logits[local:local + 1], part, geo))}
                    scores.add(row, metrics, mask.sum())
                else:
                    mask = data["gt_valid"][local].bool()
                    limits = depth_range(row["dataset"], config["data"])
                    metrics = frame_scores(output["depth"][local], data["gt"][local],
                                           mask, limits=limits)
                    scores.add(row, metrics, mask.sum())
                    if baseline is not None:
                        raw = frame_scores(data["current_depth"][local], data["gt"][local],
                                           mask, limits=limits)
                        baseline.add(row, raw, mask.sum())
            row = dataset.rows[indices[-1]]
            progress.update(len(indices), dataset=row["dataset"], sequence=row["sequence"])
    return scores.finish(), baseline.finish() if baseline is not None else None
