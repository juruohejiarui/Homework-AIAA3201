"""Development-only checkpoint selection, macro averaged by sequence then dataset."""
from collections import defaultdict

import numpy as np
import torch

from explore_a.evaluation.depth import frame_scores
from explore_a.evaluation.state import evaluation_state
from explore_a.geometry.warp import pair_geometry
from explore_a.observability.progress import Progress
from explore_a.training.losses import student_loss
from explore_a.training.prediction import predict
from explore_a.online.statistics import snapshot, difference


def validate(model, dataset, config, device, occlusion, student=None, distillation=False,
             observer=None, step=None):
    label = "DEV" if step is None else f"DEV step {step}"
    runtime = getattr(dataset, "runtime", None)
    input_before = snapshot(runtime)
    with Progress(config, label, len(dataset.rows)) as progress, evaluation_state(model, student):
        score, sources = _validate(model, dataset, config, device, occlusion, student,
                                   distillation, progress)
    if observer is not None:
        observer.last_dev_sources = sources
        observer.last_dev_input_stats = difference(runtime, input_before)
    return score


def _validate(model, dataset, config, device, occlusion, student, distillation, progress):
    scores = defaultdict(list)
    for indices, data in dataset.iter_batches():
        with torch.autocast(device_type=device, enabled=device == "cuda"
                            and config["training"]["amp"]):
            if distillation:
                geometry = pair_geometry(data, config["geometry"])
                logits = model(data, geometry)
                for local, index in enumerate(indices):
                    part = {k: v[local:local + 1] for k, v in data.items()}
                    geo = {k: v[local:local + 1] for k, v in geometry.items()}
                    if geo["hard"].any():
                        value = float(student_loss(logits[local:local + 1], part, geo))
                        row = dataset.rows[index]
                        scores[(row["dataset"], row["sequence"])].append(value)
            else:
                output, _ = predict(model, data, config, occlusion, student)
                for local, index in enumerate(indices):
                    valid = data["gt_valid"][local].bool()
                    if valid.any():
                        metrics = frame_scores(output["depth"][local], data["gt"][local],
                                               valid, all_metrics=False)
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
    return score, {source: float(np.mean(values)) for source, values in by_source.items()}
