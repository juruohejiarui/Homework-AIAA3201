"""Development reports benchmark metrics while preserving the selection arithmetic."""
import json
from pathlib import Path

import pytest
import torch

from explore_a.evaluation.depth import DEPTH_METRICS
from explore_a.observability.scalars import scalars
from explore_a.training.engine import run_training
from test_explore_a_pipeline import online_experiment, fake_backbones as fake_backbones


def test_dev_and_heldout_share_metrics_and_every_step_is_logged(tmp_path):
    config = online_experiment(tmp_path)
    run_training(config, "farneback", "rule", "flow_refine", device="cpu")
    root = Path(config["paths"]["runs"]) / "farneback-rule-flow_refine"
    records = {}
    for name in ("dev", "heldout", "training"):
        filename = "validation" if name == "dev" else name
        records[name] = [json.loads(line) for line in
                         (root / f"{filename}.jsonl").read_text().splitlines()]
    assert [row["step"] for row in records["training"]] == [1, 2]
    for row in records["training"]:
        summary = row["depth_metrics"]
        assert set(summary["macro_mean"]) == {"ssi_"+k for k in DEPTH_METRICS}
        assert summary["counts"]["valid_frames"] == 2
        tags = dict(scalars("train", row))
        assert tags["train/macro/ssi_rmse"] == summary["macro_mean"]["ssi_rmse"]
    for dev, test in zip(records["dev"], records["heldout"][1:]):
        summary = dev["depth_metrics"]
        assert set(summary["macro_mean"]) == {"ssi_"+k for k in DEPTH_METRICS}
        assert summary["macro_mean"] == pytest.approx(test["macro_mean"])
        assert summary["by_dataset"] == test["by_dataset"]
        tags = dict(scalars("dev", dev))
        assert tags["dev/macro/ssi_rmse"] == summary["macro_mean"]["ssi_rmse"]
        assert tags["dev/tum/ssi_d1"] == summary["by_dataset"]["tum"]["ssi_d1"]
        assert tags["dev/selection/ssi_abs_rel"] == dev["score"]


def test_dev_benchmark_clips_but_existing_selection_does_not(tmp_path, monkeypatch):
    config = online_experiment(tmp_path)
    from explore_a.online.factory import make_runtime
    from explore_a.online.loading import OnlinePairs
    from explore_a.data.prepare import load_manifest
    from explore_a.training.validation import validate
    from types import SimpleNamespace

    runtime = make_runtime(config, ["farneback"], False, "cpu")
    rows = [row for row in load_manifest(config)["samples"] if row["split"] == "dev"]
    dataset = OnlinePairs(config, rows, runtime, "farneback", phase="dev")
    monkeypatch.setattr("explore_a.training.validation.predict", lambda model, data, *args:
                        ({"depth": torch.ones_like(data["gt"]) * 100}, None))
    observer = SimpleNamespace()
    score = validate(torch.nn.Identity(), dataset, config, "cpu", "rule", observer=observer)
    maximum = config["data"]["indoor_max_depth_m"]
    assert score == pytest.approx(0.)
    assert observer.last_dev_metrics["macro_mean"]["ssi_rmse"] == pytest.approx(0.)
    runtime.close() if hasattr(runtime, "close") else runtime.reset()


def test_training_scores_match_benchmark_and_do_not_change_gradients(tmp_path):
    from explore_a.evaluation.depth import MetricSummary, field_scores
    from explore_a.data.reading import depth_range
    from explore_a.training.metrics import add_depth_scores, finish_depth_scores

    config = online_experiment(tmp_path)
    rows = [{"dataset": "tum", "sequence": "one", "camera": "rgb"}] * 3
    prediction = torch.tensor([[[[0., 2., 100.]]], [[[3., 4., 5.]]],
                               [[[1., 1., 1.]]]], requires_grad=True)
    target = torch.tensor([[[[1., 3., 4.]]], [[[2., 3., 4.]]], [[[0., 0., 0.]]]])
    valid = target > 0
    data = {"gt": target, "gt_valid": valid}
    actual, expected = MetricSummary(), MetricSummary()
    add_depth_scores(actual, {"depth": prediction}, data, rows, config)
    for index, row in enumerate(rows):
        expected.add(row, field_scores(prediction[index].detach(), target[index], valid[index],
                                      config, "tum"),
                     valid[index].sum())
    result = finish_depth_scores(actual)
    assert result["macro_mean"] == pytest.approx(expected.finish()["macro_mean"], rel=1e-6)
    assert result["counts"]["valid_frames"] == 2
    prediction.sum().backward()
    assert torch.equal(prediction.grad, torch.ones_like(prediction))
    empty = MetricSummary()
    add_depth_scores(empty, {"depth": prediction}, {"gt": target, "gt_valid": valid & False},
                     rows, config)
    assert finish_depth_scores(empty) is None
