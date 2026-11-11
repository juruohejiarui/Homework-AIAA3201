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
    from explore_a.evaluation.live import collect
    return collect(model, dataset, config, device, occlusion, include_baseline, step,
                   predict_fn=predict)
