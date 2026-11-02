"""Official 654-image RGB-only metric evaluation, distinct from H5 temporal held-out pairs."""
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch

from depth_project.evaluation.metrics import align_inverse_depth
from depth_project.inference.__main__ import WEIGHTS
from depth_project.io import read_rgb, write_json
from depth_project.models import load_model
from depth_project.paths import CHECKPOINTS, OUTPUTS
from explore_a.evaluation.nyu_labeled import metric_function, score


@torch.inference_mode()
def run(config, root, checkpoint, output, model):
    root, output = Path(root), Path(output)
    ids = loadmat(root / "splits.mat")["testNdxs"].ravel().astype(int).tolist()
    if ids != json.loads((root / "test/manifest.json").read_text())["test_ids"]:
        raise ValueError("Exported NYU labeled images differ from official testNdxs")
    if model == "fusion":
        from explore_a.evaluation.nyu_best import run as run_head
        return run_head(config, root, checkpoint, output / "fusion")
    adapter, records, metric = None, [], metric_function()
    for image_id in ids:
        name = f"{image_id:04d}"
        receipt = output / model / "images" / (name + ".json")
        if receipt.exists():
            records.append(json.loads(receipt.read_text()))
            continue
        truth = np.load(root / "test/gt" / (name + ".npy")).astype(np.float64)
        if model == "da2":
            saved = OUTPUTS / "first_round/da2_nyu" / (name + ".npy")
            if saved.is_file():
                inverse = np.load(saved)
            else:
                if adapter is None:
                    adapter = load_model(model, CHECKPOINTS / WEIGHTS[model], fp32=True)
                inverse = adapter.predict(read_rgb(root / "test/rgb" / (name + ".png")))
            valid = np.zeros(truth.shape, bool)
            valid[45:471, 41:601] = True
            valid &= np.isfinite(truth) & (truth > .001) & (truth < 10)
            predicted, _ = align_inverse_depth(inverse, truth, valid)
        else:
            if adapter is None:
                adapter = load_model(model, CHECKPOINTS / WEIGHTS[model], fp32=True)
            predicted = adapter.predict(read_rgb(root / "test/rgb" / (name + ".png")))
        if predicted.shape != truth.shape or not np.isfinite(predicted).all():
            raise ValueError(f"Invalid NYU prediction: {name}")
        record = {"image_id": image_id, **score(predicted, truth, metric)}
        write_json(receipt, record)
        records.append(record)
        if len(records) % 100 == 0:
            print("NYU", model, len(records), "/", len(ids), flush=True)
    keys = set(records[0]) - {"image_id"}
    result = {"model": model, "count": len(records), "test_ids": ids,
              "mean": {k: float(np.mean([r[k] for r in records])) for k in sorted(keys)},
              "alignment": "GT-assisted inverse SSI" if model == "da2" else "none",
              "protocol": "official 654; RGB-only; crop[45:471,41:601]; .001-10m; image mean",
              "precision": "DA2 reused predictions follow saved profile; new inference FP32",
              "per_image": records}
    write_json(output / model / "metrics.json", result)
    return result
