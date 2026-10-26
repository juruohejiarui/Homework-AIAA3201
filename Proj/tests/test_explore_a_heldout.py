"""Test monitoring is observational: no training, selection or random-state side effects."""
import copy
import csv
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from depth_project.io import read_json
from explore_a.evaluation.depth import DEPTH_METRICS, MetricSummary
from explore_a.evaluation.heldout import collect_heldout
from explore_a.evaluation.state import evaluation_state
from explore_a.training.checkpoint import seed_everything
from explore_a.training.engine import run_training
from explore_a.training.monitor import HeldoutMonitor, scheduled
from test_explore_a_pipeline import online_experiment, fake_backbones as fake_backbones


def read_curve(root):
    return [json.loads(line) for line in (root / "heldout.jsonl").read_text().splitlines()]


def test_schedule_includes_initial_and_nonmultiple_final_and_can_disable():
    assert [s for s in range(8) if scheduled(s, 7, 3)] == [0, 3, 6, 7]
    assert not any(scheduled(s, 7, 0) for s in range(8))


def test_evaluation_restores_nested_modes_rng_parameters_and_gradients_on_error():
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Dropout(.5),
                                torch.nn.Linear(3, 1)).train()
    model[0].eval()
    modes = [module.training for module in model.modules()]
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    seed_everything(42)
    expected = random.random(), np.random.rand(), torch.rand(1)
    seed_everything(42)
    with pytest.raises(RuntimeError, match="interrupted"):
        with evaluation_state(model):
            assert not any(module.training for module in model.modules())
            assert not model(torch.ones(2, 3)).requires_grad
            random.random(), np.random.rand(), torch.rand(5)
            raise RuntimeError("interrupted")
    assert [module.training for module in model.modules()] == modes
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(1), expected[2])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[key])
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters())


def test_monitor_rejects_training_rows_before_reading_data():
    data = SimpleNamespace(rows=[{"split": "train"}])
    with pytest.raises(ValueError, match="only test rows"):
        collect_heldout(None, data, None, "cpu", "rule")


def test_macro_mean_weights_sequences_and_sources_equally_not_pixel_counts():
    result = MetricSummary()
    for source, sequence, value, pixels in (("a", "one", 1, 10), ("a", "one", 3, 1),
                                            ("a", "two", 8, 1), ("b", "one", 100, 10)):
        result.add({"dataset": source, "sequence": sequence, "camera": "rgb"},
                   {"abs_rel": value}, pixels)
    summary = result.finish()
    assert summary["by_dataset"]["a"]["abs_rel"] == 5
    assert summary["macro_mean"]["abs_rel"] == 52.5
    assert summary["counts"] == {"total_frames": 4, "valid_frames": 4, "valid_pixels": 22}


def test_monitor_cannot_change_updates_or_best_development_checkpoint(tmp_path, monkeypatch):
    config = online_experiment(tmp_path)
    original_runs = config["paths"]["runs"]
    finished = []
    for enabled in (False, True):
        current = copy.deepcopy(config)
        current["paths"]["runs"] = original_runs + ("-monitored" if enabled else "-control")
        current["training"]["heldout_every"] = 1 if enabled else 0
        development_scores = iter([.1, .2])
        monkeypatch.setattr("explore_a.training.engine.validate",
                            lambda *args, **kwargs: next(development_scores))
        run_training(current, "raft", "rule", "residual", device="cpu")
        root = Path(current["paths"]["runs"]) / "raft-rule-residual"
        last = torch.load(root / "last.pt", weights_only=False)
        best = torch.load(root / "best.pt", weights_only=False)
        assert best["step"] == 1 and best["best"] == .1
        finished.append(last)
        if enabled:
            records = read_curve(root)
            assert [record["step"] for record in records] == [0, 1, 2]
            assert all(record["split"] == "test" for record in records)
            assert all(set(record["macro_mean"]) == set(DEPTH_METRICS) for record in records)
            assert all(record["counts"]["valid_frames"] == 2 for record in records)
            assert all({s["sequence"] for s in record["sequences"]} == {"test"}
                       for record in records)
            baseline = read_json(root / "heldout_baseline.json")
            np.testing.assert_allclose(baseline["macro_mean"]["abs_rel"], 1 / 3, rtol=1e-6)
            for record in records:
                for metric in DEPTH_METRICS:
                    assert record["delta_macro_mean"][metric] == (
                        record["macro_mean"][metric] - baseline["macro_mean"][metric])
            with (root / "heldout.csv").open() as stream:
                table = list(csv.DictReader(stream))
            assert [int(row["step"]) for row in table if row["dataset"] == "__macro__"] \
                == [0, 1, 2]
        else:
            assert not (root / "heldout.jsonl").exists()
    for key, value in finished[0]["model"].items():
        torch.testing.assert_close(value, finished[1]["model"][key], rtol=0, atol=0)
    torch.testing.assert_close(finished[0]["torch_rng"], finished[1]["torch_rng"])
    assert finished[0]["scheduler"] == finished[1]["scheduler"]
    assert finished[0]["identity"] == finished[1]["identity"]
    # Enabling observation on a completed older run does not invalidate its checkpoint.
    legacy = copy.deepcopy(config)
    legacy["paths"]["runs"] = original_runs + "-control"
    root = Path(legacy["paths"]["runs"]) / "raft-rule-residual"
    before = (root / "last.pt").stat().st_mtime_ns
    run_training(legacy, "raft", "rule", "residual", device="cpu")
    assert (root / "last.pt").stat().st_mtime_ns == before
    assert [record["step"] for record in read_curve(root)] == [2]


def test_resume_keeps_curves_without_duplicate_steps_or_reselecting_best(tmp_path, monkeypatch):
    config = online_experiment(tmp_path)
    config["training"].update(steps=3, heldout_every=2)
    original = HeldoutMonitor.maybe_run
    stop = [True]
    def interrupt_after_record(self, model, step):
        original(self, model, step)
        if step == 2 and stop[0]:
            stop[0] = False
            raise RuntimeError("simulated interruption after test logging")
    monkeypatch.setattr(HeldoutMonitor, "maybe_run", interrupt_after_record)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_training(config, "raft", "rule", "unet", device="cpu")
    root = Path(config["paths"]["runs"]) / "raft-rule-unet"
    first = read_curve(root)
    assert [record["step"] for record in first] == [0, 2]
    # A partial last write and missing CSV are recoverable without losing valid snapshots.
    with (root / "heldout.jsonl").open("a") as stream:
        stream.write('{"step":')
    (root / "heldout.csv").unlink()
    run_training(config, "raft", "rule", "unet", device="cpu")
    records = read_curve(root)
    assert [record["step"] for record in records] == [0, 2, 3]
    assert records[:2] == first
    size = (root / "heldout.jsonl").stat().st_size
    run_training(config, "raft", "rule", "unet", device="cpu")
    assert (root / "heldout.jsonl").stat().st_size == size


def test_student_reports_only_distillation_bce(tmp_path):
    config = online_experiment(tmp_path)
    run_training(config, "raft", distillation=True, device="cpu")
    root = Path(config["paths"]["runs"]) / "occlusion-raft"
    records = read_curve(root)
    assert [record["step"] for record in records] == [0, 1, 2]
    assert all(set(record["macro_mean"]) == {"heldout_bce"} for record in records)
    assert all(record["task"] == "mftiq_soft_label_distillation" for record in records)
    assert not (root / "heldout_baseline.json").exists()
