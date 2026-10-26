"""Tiny raw sequences verify real decoding, online training, evaluation and checkpoint resume."""
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from depth_project.io import read_json, write_json
from explore_a.config import load_config
from explore_a.data.history import evaluation_history
from explore_a.evaluation.depth import DEPTH_METRICS
from explore_a.evaluation.engine import evaluate
from explore_a.training.engine import run_training


@pytest.fixture(autouse=True)
def fake_backbones(monkeypatch):
    class Depth:
        model = torch.nn.Identity()
        calls = 0

        def __init__(self, *_, **__):
            pass

        def predict(self, rgb):
            type(self).calls += 1
            return np.ones(rgb.shape[:2], np.float32) * 2

    class Flow:
        def __init__(self, *_):
            pass

        def __call__(self, previous, current):
            field = np.zeros((2, *current.shape[:2]), np.float32)
            return field, field.copy()

    class Teacher:
        def __init__(self, *_):
            pass

        def __call__(self, previous, current, backward):
            return {"visibility": np.ones((1, *current.shape[:2]), np.float32) * .8}

    monkeypatch.setattr("explore_a.online.frozen.DepthProAdapter", Depth)
    monkeypatch.setattr("explore_a.online.frozen.PairFlow", Flow)
    monkeypatch.setattr("explore_a.online.frozen.MFTIQVisibility", Teacher)
    monkeypatch.setattr("explore_a.online.frozen.check_weights", lambda *_, **__: None)
    return Depth, Flow, Teacher


def online_experiment(tmp_path):
    torch.set_num_threads(1)
    config = load_config()
    config["observability"].update(progress="off", tensorboard=False, plots=False)
    for name in config["weights"]:
        path = tmp_path / (name + ".dummy")
        path.write_bytes(b"identity only; models are not loaded")
        config["weights"][name] = str(path)
    config["paths"].update(manifest=str(tmp_path / "data.json"), runs=str(tmp_path / "runs"))
    config["data"]["canvas_indoor"] = [16, 24]
    config["training"].update(steps=2, microbatch=1, effective_batch=2, workers=0,
                              validate_every=1, save_every=1, log_every=1, amp=False,
                              heldout_every=1)
    config["evaluation"]["visualizations_per_sequence"] = 0
    rows = []
    for split in ("train", "dev", "test"):
        folder = tmp_path / "raw" / split
        folder.mkdir(parents=True)
        for frame in range(3):
            assert cv2.imwrite(str(folder / f"rgb{frame}.png"),
                               np.ones((16, 24, 3), np.uint8) * 100)
            assert cv2.imwrite(str(folder / f"depth{frame}.png"),
                               np.ones((16, 24), np.uint16) * 15000)
        for frame in (1, 2):
            rows.append({"id": f"tum:{split}:rgb:{frame}", "dataset": "tum", "sequence": split,
                         "camera": "rgb", "group": split, "split": split,
                         "previous_id": str(frame - 1), "current_id": str(frame),
                         "current_rgb": str(folder / f"rgb{frame}.png"),
                         "previous_rgb": str(folder / f"rgb{frame-1}.png"),
                         "gt": str(folder / f"depth{frame}.png"),
                         "previous_gt": str(folder / f"depth{frame-1}.png"),
                         "depth_encoding": "tum_png", "K": None, "pose": None})
    write_json(Path(config["paths"]["manifest"]), {"schema": 1, "settings": config["data"],
               "history": evaluation_history(), "samples": rows, "test_label": "fixture"})
    return config


def test_online_training_resume_and_temporal_evaluation(tmp_path):
    config = online_experiment(tmp_path)
    run_training(config, "raft", distillation=True, device="cpu")
    run_training(config, "raft", "student", "residual", device="cpu")
    last = tmp_path / "runs/raft-student-residual/last.pt"
    before = last.stat().st_mtime_ns
    run_training(config, "raft", "student", "residual", device="cpu")
    assert last.stat().st_mtime_ns == before
    result = evaluate(config, "raft", "student", "residual", device="cpu")
    assert result["sequences"][0]["coverage"]["consecutive_outputs"] == 1
    assert "gt_corrected_temporal_mae_m" in result["by_dataset"]["tum"]
    raw = evaluate(config, "farneback", baseline="raw", device="cpu")
    np.testing.assert_allclose(raw["by_dataset"]["tum"]["abs_rel"], 1 / 3, rtol=1e-6)
    assert raw["latency"]["mode"] == "independent_online"
    assert raw["latency"]["warm_ms"] > 0
    state = read_json(last.parent / "complete.json")
    assert state["steps"] == 2
    baseline = read_json(last.parent / "heldout_baseline.json")
    for metric in DEPTH_METRICS:
        np.testing.assert_allclose(baseline["macro_mean"][metric], raw["macro_mean"][metric])
    assert not list(tmp_path.rglob("*.npz"))
