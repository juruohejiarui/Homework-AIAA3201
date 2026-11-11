"""Depth-only external references on exactly the fusion evaluation endpoints and canvas."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from depth_project.evaluation.metrics import align_inverse_depth
from depth_project.inference.__main__ import WEIGHTS
from depth_project.io import write_json
from depth_project.models import load_model
from depth_project.paths import CHECKPOINTS
from explore_a.data.reading import depth_range, fit_depth, fit_image, read_frame
from explore_a.evaluation.accumulator import average
from explore_a.evaluation.depth import frame_scores


def canvas_prediction(prediction, target, valid, canvas, kind):
    """Relative DA-V2 is GT-aligned, and must never be presented as metric RGB-only."""
    fitted, _ = fit_image(np.asarray(prediction, np.float32), canvas)
    if not np.isfinite(fitted).all():
        raise ValueError("Non-finite reference prediction")
    if kind == "relative_inverse":
        if valid.any():
            fitted, _ = align_inverse_depth(fitted, target, valid)
    elif kind == "metric_m":
        if (fitted[valid] <= 0).any():
            raise ValueError("Nonpositive metric depth")
    else:
        raise ValueError(f"Unsupported reference kind: {kind}")
    return fitted


@torch.inference_mode()
def run(config, rows, model, output):
    output = Path(output)
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"], row["sequence"], row["camera"]].append(row)
    adapter, records = None, []
    for index, (key, selected) in enumerate(groups.items()):
        receipt = output / "sequences" / f"{index:04d}.json"
        ids = [r["id"] for r in selected]
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if record["sample_ids"] != ids or record.get("scoring") != "dual-v1":
                raise ValueError("Reference receipt has different frame IDs")
            records.append(record)
            continue
        from explore_a.evaluation.alignment import RelativeScores
        aligned = RelativeScores()
        scores, valid_frames = [], 0
        for number, row in enumerate(selected):
            rgb, gt = read_frame(row)
            canvas = config["data"]["canvas_kitti" if row["dataset"] == "kitti"
                                     else "canvas_indoor"]
            minimum, maximum = depth_range(row["dataset"], config["data"])
            target = np.zeros(canvas, np.float32) if gt is None else fit_depth(gt, canvas)
            valid = np.isfinite(target) & (target > minimum) & (target < maximum)
            if valid.any():
                if adapter is None:
                    adapter = load_model(model, CHECKPOINTS / WEIGHTS[model], fp32=True)
                raw_prediction = adapter.predict(rgb)
                if adapter.kind == "relative_inverse":
                    field, _ = fit_image(np.asarray(raw_prediction, np.float32), canvas)
                    scores.append(aligned.add(row, field, target, valid, (minimum, maximum)))
                prediction = canvas_prediction(raw_prediction, target, valid,
                                               canvas, adapter.kind)
                scores.append(frame_scores(torch.from_numpy(prediction),
                                           torch.from_numpy(target), torch.from_numpy(valid),
                                           (minimum, maximum)))
                valid_frames += 1
            if (number + 1) % 50 == 0:
                print("REFERENCE", model, key[:2], number + 1, "/", len(selected), flush=True)
        for row, values, count in aligned.finish():
            scores.append(values)
        record = {"scoring": "dual-v1", "dataset": key[0], "sequence": key[1], "camera": key[2],
                  "sample_ids": ids, "metrics": average(scores),
                  "coverage": {"pairs": len(selected), "valid_gt_frames": valid_frames}}
        write_json(receipt, record)
        records.append(record)
    sources = sorted({r["dataset"] for r in records})
    by_dataset = {s: average([r["metrics"] for r in records if r["dataset"] == s])
                  for s in sources}
    result = {"model": model, "precision": "fp32", "sequences": records,
              "by_dataset": by_dataset, "macro_mean": average(list(by_dataset.values())),
              "protocol": "Same current endpoints, canvas, GT splat, range and sequence macro",
              "alignment": "per-frame GT-assisted inverse SSI" if model == "da2" else "none",
              "temporal_note": "Depth metrics only; no temporal scores or standalone latency"}
    write_json(output / "metrics.json", result)
    return result
