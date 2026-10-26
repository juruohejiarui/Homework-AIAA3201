"""Streaming per-experiment metrics and temporal state, without retaining dense predictions."""
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from explore_a.data.reading import depth_range
from explore_a.evaluation.depth import frame_scores
from explore_a.evaluation.output import safe_id
from explore_a.evaluation.temporal import flow_temporal, geometric_temporal


def average(rows):
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}


class EvaluationAccumulator:
    def __init__(self, config, name, split, sink):
        self.config, self.name, self.split = config, name, split
        self.root = Path(config["paths"]["runs"]) / name / f"evaluation-{split}"
        self.sink = sink
        sink.open(self.root)
        self.sums = defaultdict(lambda: defaultdict(float))
        self.counts = defaultdict(lambda: defaultdict(int))
        self.coverage = defaultdict(lambda: {"pairs": 0, "consecutive_outputs": 0,
                                            "valid_gt_frames": 0})
        self.visual_counts = defaultdict(int)
        self.previous = self.previous_key = self.first_median = None
        self.timings, self.head_timings = [], []

    def add(self, row, data, output, scoring):
        key = row["dataset"], row["sequence"], row["camera"]
        _, maximum = depth_range(row["dataset"], self.config["data"])
        prediction = output["depth"].float().clamp(*depth_range(
            row["dataset"], self.config["data"]))
        valid = data["gt_valid"][0, 0].bool().cpu().numpy()
        record = {}
        self.coverage[key]["pairs"] += 1
        if valid.any():
            record.update(frame_scores(prediction[0, 0], data["gt"][0, 0],
                                       data["gt_valid"][0, 0]))
            eligible = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            record["boundary_valid_fraction"] = float(eligible.mean())
            if eligible.any():
                boundary = boundary_score(prediction[0, 0].cpu().numpy(),
                                          data["gt"][0, 0].cpu().numpy(), valid, self.config)
                record["custom_boundary_f1"] = boundary["f1"]
            self.coverage[key]["valid_gt_frames"] += 1
        if self.previous_key == (*key, row["previous_id"]):
            record.update(flow_temporal(prediction, self.previous, scoring,
                                        self.config["geometry"]))
            record.update(geometric_temporal(prediction, self.previous, scoring))
            self.coverage[key]["consecutive_outputs"] += 1
        if self.previous_key is None or self.previous_key[:3] != key:
            self.first_median = prediction[data["image_valid"].bool()].median().item()
        record["median_scale_ratio_to_first"] = float(
            prediction[data["image_valid"].bool()].median() / max(self.first_median, 1e-6))
        self.previous, self.previous_key = prediction.detach(), (*key, row["current_id"])
        for metric, value in record.items():
            self.sums[key][metric] += value
            self.counts[key][metric] += 1
        self.sink.record(self.root, {"sample": row["id"], **record})
        if self.visual_counts[key] < self.config["evaluation"]["visualizations_per_sequence"]:
            self.sink.panel(self.root / "visualizations" / (safe_id(row["id"]) + ".jpg"),
                            data, {**output, "depth": prediction}, maximum)
            self.visual_counts[key] += 1

    def finish(self, manifest, latency):
        sequences = [{"dataset": key[0], "sequence": key[1], "camera": key[2],
                      "metrics": {name: total / self.counts[key][name]
                                  for name, total in self.sums[key].items()},
                      "coverage": coverage} for key, coverage in sorted(self.coverage.items())]
        by_dataset = {source: average([s["metrics"] for s in sequences if s["dataset"] == source])
                      for source in sorted({s["dataset"] for s in sequences})}
        return {"experiment": self.name, "split": self.split, "test_label": manifest["test_label"],
                "protocol": "online metric meters, no GT scale alignment; padded canvas",
                "temporal_note": "Consecutive evaluated RGB outputs only; GT-corrected motion "
                                 "residual when two measured depths exist; not paper 3D TAE.",
                "boundary_note": "Custom log-depth Sobel F1; omitted without valid 3x3 GT.",
                "sequences": sequences, "by_dataset": by_dataset,
                "macro_mean": average(list(by_dataset.values())), "latency": latency}


def boundary_score(predicted, target, valid, config):
    from depth_project.evaluation.metrics import boundary_score as score
    return score(predicted, target, valid, config["evaluation"]["boundary_threshold"],
                 config["evaluation"]["boundary_tolerance"])
