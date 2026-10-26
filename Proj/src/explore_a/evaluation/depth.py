"""Shared frame scoring and sequence/dataset macro aggregation for metric depth."""
from collections import defaultdict

import numpy as np

from depth_project.evaluation.metrics import depth_metrics

DEPTH_METRICS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "d1", "d2", "d3")


def frame_scores(prediction, target, valid, limits=None, all_metrics=True):
    """Score a HW tensor on measured GT; limits match the final evaluation protocol."""
    valid = valid.bool()
    if not valid.any():
        return None
    prediction = prediction.float()
    if limits is not None:
        prediction = prediction.clamp(*limits)
    if not all_metrics:
        # Preserve the original development selection arithmetic and unclipped prediction.
        value = ((prediction - target).abs() / target.clamp_min(1e-6))[valid].mean()
        return {"abs_rel": float(value)}
    return depth_metrics(prediction[valid].cpu().numpy(), target[valid].float().cpu().numpy())


def metric_means(rows):
    return {key: float(np.mean([row[key] for row in rows if key in row]))
            for key in sorted({name for row in rows for name in row})}


class MetricSummary:
    """Equal frame weight within camera-sequence, then equal sequence and dataset weight."""
    def __init__(self):
        self.metrics = defaultdict(list)
        self.counts = defaultdict(lambda: {"total_frames": 0, "valid_frames": 0,
                                           "valid_pixels": 0})

    def add(self, row, metrics, valid_pixels):
        key = (row["dataset"], row["sequence"], row["camera"])
        self.counts[key]["total_frames"] += 1
        if metrics is not None:
            self.metrics[key].append(metrics)
            self.counts[key]["valid_frames"] += 1
            self.counts[key]["valid_pixels"] += int(valid_pixels)

    def finish(self):
        sequences = [{"dataset": key[0], "sequence": key[1], "camera": key[2],
                      "metrics": metric_means(self.metrics[key]), "counts": counts}
                     for key, counts in sorted(self.counts.items())]
        by_dataset, dataset_counts = {}, {}
        for source in sorted({row["dataset"] for row in sequences}):
            selected = [row for row in sequences if row["dataset"] == source]
            by_dataset[source] = metric_means([row["metrics"] for row in selected])
            dataset_counts[source] = {key: sum(row["counts"][key] for row in selected)
                                      for key in ("total_frames", "valid_frames", "valid_pixels")}
        if not any(by_dataset.values()):
            raise ValueError("No valid held-out supervision")
        counts = {key: sum(row[key] for row in dataset_counts.values())
                  for key in ("total_frames", "valid_frames", "valid_pixels")}
        return {"sequences": sequences, "by_dataset": by_dataset,
                "dataset_counts": dataset_counts, "macro_mean": metric_means(
                    list(by_dataset.values())), "counts": counts}
