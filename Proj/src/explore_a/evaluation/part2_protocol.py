"""Part 2 manifest paths, canvas restoration and unmodified upstream SSI scoring."""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from depth_project.evaluation.official import load_evaluator
from explore_a.data.reading import canvas_transform


def benchmark_path(value, root):
    """Resolve old storage spelling under the benchmark root, never the training root."""
    path = Path(value)
    for marker in ("/Datasets/", "/datasets/"):
        if marker in str(path):
            candidate = Path(root) / str(path).split(marker, 1)[1]
            if candidate.resolve().is_relative_to(Path(root).resolve()) and candidate.is_file():
                return str(candidate)
            raise FileNotFoundError(candidate)
    if path.resolve().is_relative_to(Path(root).resolve()) and path.is_file():
        return str(path)
    raise FileNotFoundError(path)


def restore_canvas(value, image_shape, canvas):
    size, left, top = canvas_transform(image_shape, canvas)
    crop = value[top:top + size[1], left:left + size[0]]
    return cv2.resize(crop, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_LINEAR)


def official_score(inverse_depths, frames, config):
    """Adapt only prediction I/O; upstream least squares and metrics remain unchanged."""
    evaluator = load_evaluator(device="cpu")
    tokens = {f"memory-{i}.npy": value for i, value in enumerate(inverse_depths)}
    paths = list(tokens)
    original_os, original_infer = evaluator.os, evaluator.get_infer

    def get_infer(path, args, target_size=None):
        value = tokens[path].astype(np.float32)
        if target_size is not None and value.shape != tuple(target_size):
            value = cv2.resize(value, (target_size[1], target_size[0]))
        return value

    # Replace this fresh module's os reference, not the process-wide os.path function.
    evaluator.os = SimpleNamespace(path=SimpleNamespace(exists=lambda p: p in tokens))
    evaluator.get_infer = get_infer
    top, bottom, left, right = config["crop"]
    args = SimpleNamespace(a=top, b=bottom, c=left, d=right,
                           max_eval_len=len(frames), max_depth_eval=config["max_depth"])
    try:
        result = evaluator.eval_depthcrafter(paths, [r["gt"] for r in frames],
                                            [config["factor"]] * len(frames), args)
    finally:
        # Do not leave sequence arrays retained by an imported module's function cycle.
        evaluator.os, evaluator.get_infer = original_os, original_infer
    return dict(zip(("abs_rel", "rmse", "delta1"), map(float, result)))


def baseline_sequence_scores(branches, frames, config):
    """One baseline SSI gauge for the whole video, reused unchanged by every improved branch.

    This is a separate diagnostic; it does not replace the published evaluator's refit.
    """
    from depth_project.evaluation.metrics import depth_metrics
    evaluator = load_evaluator(device="cpu")
    top, bottom, left, right = config["crop"]
    targets, baseline, masks = [], [], []
    for value, frame in zip(branches["raw_canvas"], frames):
        target = evaluator.get_gt(frame["gt"], config["factor"], None)[top:bottom, left:right]
        targets.append(target)
        baseline.append(cv2.resize(value, (target.shape[1], target.shape[0])))
        masks.append((target > 1e-3) & (target < config["max_depth"]))
    x = np.concatenate([p[m] for p, m in zip(baseline, masks)]).astype(np.float64)
    y = np.concatenate([1/g[m] for g, m in zip(targets, masks)]).astype(np.float64)
    if not len(x):
        return {"gauge": None, "branches": {}}
    variance = np.mean((x-x.mean())**2)
    a = max(0., float(np.mean((x-x.mean())*(y-y.mean()))/variance)) if variance > 1e-15 else 0.
    b = float(y.mean()-a*x.mean())
    result = {}
    for name, predictions in branches.items():
        rows = []
        for value, target, valid in zip(predictions, targets, masks):
            inverse = cv2.resize(value, (target.shape[1], target.shape[0]))
            depth = np.clip(1/np.maximum(a*inverse+b, 1e-6), 1e-3, config["max_depth"])
            if valid.any():
                rows.append(depth_metrics(depth[valid], target[valid]))
        result[name] = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
    return {"gauge": {"scale": a, "shift": b, "anchor": "raw_canvas, whole sequence"},
            "branches": result}


def metric_absrel(inverse_depths, frames, config):
    """Secondary metric-scale diagnostic; official SSI results are reported separately."""
    evaluator = load_evaluator(device="cpu")
    top, bottom, left, right = config["crop"]
    values = []
    for prediction, frame in zip(inverse_depths, frames):
        target = evaluator.get_gt(frame["gt"], config["factor"], None)
        target = target[top:bottom, left:right]
        prediction = cv2.resize(prediction, (target.shape[1], target.shape[0]))
        depth = np.clip(1 / np.maximum(prediction, 1e-8), 1e-3, config["max_depth"])
        valid = (target > 1e-3) & (target < config["max_depth"])
        if valid.any():
            values.append(float(np.mean(np.abs(depth[valid] - target[valid]) / target[valid])))
    return float(np.mean(values)) if values else None


def metric_scores(inverse_depths, frames, config):
    """Unaligned metric depth scores; GT is used only after prediction, never for scaling."""
    from depth_project.evaluation.metrics import depth_metrics

    evaluator = load_evaluator(device="cpu")
    top, bottom, left, right = config["crop"]
    records = []
    for prediction, frame in zip(inverse_depths, frames):
        target = evaluator.get_gt(frame["gt"], config["factor"], None)[top:bottom, left:right]
        inverse = cv2.resize(prediction, (target.shape[1], target.shape[0]))
        depth = np.clip(1 / np.maximum(inverse, 1e-8), 1e-3, config["max_depth"])
        valid = np.isfinite(target) & (target > 1e-3) & (target < config["max_depth"])
        if valid.any():
            records.append(depth_metrics(depth[valid], target[valid]))
    return {key: float(np.mean([r[key] for r in records])) for key in records[0]} if records else {}


def aligned_sequence_scores(predictions, frames, config, sample_prefix):
    """Additional image alignment and raw ordering beside the upstream video protocol."""
    from explore_a.evaluation.alignment import RelativeScores, ordinal
    from explore_a.evaluation.accumulator import average
    from explore_a.evaluation.sintel_ac import PROTOCOL, IMAGE_METRICS, image_scores
    ac = config.get("image_scoring") == PROTOCOL
    evaluator = load_evaluator(device="cpu")
    top, bottom, left, right = config["crop"]
    scorer, records = RelativeScores(), []
    try:
        for i, (prediction, frame) in enumerate(zip(predictions, frames)):
            target = evaluator.get_gt(frame["gt"], config["factor"], None)[top:bottom, left:right]
            value = cv2.resize(prediction, (target.shape[1], target.shape[0]))
            valid = np.isfinite(target) & (target > 1e-3) & (target < config["max_depth"])
            row = {"id": f"{sample_prefix}:{i}", "dataset": "benchmark",
                   "sequence": sample_prefix, "camera": "video"}
            # Match the upstream evaluator's input floor before inverse-depth fitting.
            measured = scorer.add(row, np.maximum(value, 1e-3), target, valid,
                                  (1e-3, config["max_depth"]))
            if ac:
                measured = {k: v for k, v in measured.items() if k not in IMAGE_METRICS}
                measured.update(image_scores(value, target))
            measured.update(ordinal(value[valid], target[valid], row["id"]))
            records.append(measured)
        for row, values, count in scorer.finish():
            records.append(values)
        return average(records)
    finally:
        scorer.close()
