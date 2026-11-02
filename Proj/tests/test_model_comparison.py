"""CPU-only protocol checks; no pretrained inference or benchmark execution."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from depth_project.io import write_json
from explore_a.evaluation import comparison
from explore_a.evaluation.comparison_report import report
from explore_a.evaluation.reference_splits import canvas_prediction
from explore_a.training.dev_subset import select_dev


def test_same_fixed_dev_cohort_and_full_test(monkeypatch):
    rows = [{"id": f"{part}:{i}", "dataset": "nyuv2", "sequence": f"room{i % 2}",
             "split": part} for part in ("dev", "test") for i in range(100)]
    monkeypatch.setattr(comparison, "load_manifest", lambda c: {"samples": rows})
    config = {"training": {"dev_sample_fraction": .1, "seed": 2026}}
    dev = [r for r in rows if r["split"] == "dev"]
    assert comparison.cohort(config, "dev") == select_dev(dev, .1, 2026)
    assert len(comparison.cohort(config, "dev")) == 10
    assert len(comparison.cohort(config, "dev", True)) == 100
    assert len(comparison.cohort(config, "test")) == 100


def test_metric_depth_is_not_gt_aligned_but_relative_reference_is():
    target = np.linspace(1, 8, 48).astype(np.float32).reshape(6, 8)
    valid = np.ones_like(target, bool)
    metric = canvas_prediction(target * 2, target, valid, target.shape, "metric_m")
    np.testing.assert_allclose(metric, target * 2)
    inverse = .7 / target + .04
    aligned = canvas_prediction(inverse, target, valid, target.shape, "relative_inverse")
    np.testing.assert_allclose(aligned, target, rtol=1e-5)
    with pytest.raises(ValueError, match="Nonpositive"):
        canvas_prediction(-target, target, valid, target.shape, "metric_m")


def test_snapshot_does_not_follow_best_updates(tmp_path):
    source = tmp_path / "best.pt"
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    config = {"paths": {"manifest": str(manifest)}, "data": {}, "geometry": {},
              "training": {"dev_sample_fraction": .1, "seed": 2026}}
    state = {"model": {"w": torch.ones(1)}, "identity": {}, "step": 28, "best": .1,
             "optimizer": {"not_for_evaluation": True}}
    torch.save(state, source)
    snapshot = comparison.pin(config, source, tmp_path / "result")
    state["step"] = 30
    torch.save(state, source)
    comparison.pin(config, source, tmp_path / "result")
    saved = torch.load(snapshot, weights_only=False)
    assert saved["step"] == 28 and "optimizer" not in saved


def test_unidepth_benchmark_converts_metric_depth_to_inverse(monkeypatch, tmp_path):
    from explore_a.evaluation import part2_references as ref

    manifest = {"config": {"crop": [0, 4, 0, 6]}, "sequences": [
        {"name": "sequence", "frames": [{"image": "rgb.png", "gt": "gt.npy"}]}]}
    write_json(tmp_path / ref.MANIFESTS["bonn"], manifest)
    monkeypatch.setattr(ref, "select_sequences", lambda m, s: m["sequences"])
    monkeypatch.setattr(ref, "benchmark_path", lambda p, r: p)
    monkeypatch.setattr(ref, "input_rgb", lambda *a: np.zeros((4, 6, 3), np.uint8))
    monkeypatch.setattr(ref, "load_model", lambda *a, **kw: SimpleNamespace(kind="metric_m"))
    monkeypatch.setattr(ref, "predict", lambda *a: (np.full((1, 4, 6), 2.), {}))

    def score(prediction, *args):
        np.testing.assert_allclose(prediction, .5)
        return {"abs_rel": 0., "rmse": 0., "delta1": 1.}

    monkeypatch.setattr(ref, "official_score", score)
    monkeypatch.setattr(ref, "metric_absrel", lambda *a: .125)
    result = ref.run("unidepthv2", tmp_path / "out", tmp_path, ["bonn"])
    assert result["bonn"]["sequences"][0]["metric_absrel_secondary"] == .125
    assert not list((tmp_path / "out").rglob("*.npz"))


def test_report_keeps_metric_and_ssi_protocols_visible(tmp_path):
    values = {"abs_rel": .1, "rmse": .2, "d1": .9}
    write_json(tmp_path / "dev/da2/metrics.json", {
        "by_dataset": {"nyuv2": values}, "alignment": "per-frame GT-assisted inverse SSI"})
    write_json(tmp_path / "nyu/unidepthv2/metrics.json", {
        "mean": values, "alignment": "none"})
    rows = report(tmp_path)
    assert {r["scope"] for r in rows} == {"dev", "official-labeled"}
    assert "GT-assisted" in (tmp_path / "comparison.md").read_text()
    assert len(json.loads((tmp_path / "comparison.json").read_text())) == 2
