"""AC's far-GT handling, unconstrained fit, cohort and dataset-level averaging."""
import copy
import json

import numpy as np
import pytest

from explore_a.evaluation.sintel_ac import PROTOCOL, IMAGE_METRICS, image_scores, final_rgb, prepare_manifest
from explore_a.evaluation.session import CaseScores, VERSION, summarize


def test_inverse_fit_can_have_negative_scale_and_keeps_far_gt():
    truth = np.array([[1., 2., 4., 100.]], np.float32)
    prediction = 2 - 1 / np.minimum(truth, 72)
    measured = image_scores(prediction, truth)
    assert measured["absrel_img"] < 2e-4
    assert measured["delta1_img"] == 1.


def test_case_scores_include_frames_with_no_video_gt():
    truth = np.full((3, 4), 200., np.float32)
    row = {"id": "sintel:far:0", "dataset": "sintel", "sequence": "far"}
    scorer = CaseScores(sintel_ac=True)
    try:
        scorer.add("baseline", row, np.ones_like(truth), truth,
                   np.zeros_like(truth, bool), (1e-3, 70))
        result = scorer.finish()["baseline"]
        # A singular inverse fit yields the public evaluator's 80 m prediction;
        # GT is clipped to 72 m, rather than discarding this frame.
        assert result["absrel_img"] == pytest.approx(8 / 72)
        assert result["delta1_img"] == 1.
        assert "absrel_video" not in result
    finally:
        scorer.close()


def test_final_pass_is_required_and_partial_cohort_is_rejected(tmp_path):
    clean = tmp_path / "training/clean/alley/frame.png"
    clean.parent.mkdir(parents=True)
    clean.touch()
    with pytest.raises(FileNotFoundError, match="final RGB"):
        final_rgb(clean)
    final = tmp_path / "training/final/alley/frame.png"
    final.parent.mkdir(parents=True)
    final.touch()
    assert final_rgb(clean) == str(final)
    with pytest.raises(ValueError, match="1064"):
        prepare_manifest({"dataset": "sintel", "sequences": [{"frames": [{}]}]}, tmp_path)


def test_manifest_selects_all_final_frames_without_mutating_other_datasets(tmp_path, monkeypatch):
    import explore_a.evaluation.sintel_ac as ac
    monkeypatch.setattr("explore_a.evaluation.part2_protocol.benchmark_path", lambda p, root: p)
    monkeypatch.setattr(ac, "final_rgb", lambda p, root=None: p.replace("/clean/", "/final/"))
    manifest = {"dataset": "sintel", "config": {}, "sequences": [
        {"name": str(i), "frames": [{"image": "training/clean/a.png"} for _ in range(n)]}
        for i, n in enumerate([50] * 21 + [7, 7])]}
    prepare_manifest(manifest, tmp_path)
    assert sum(len(s["frames"]) for s in manifest["sequences"]) == 1064
    assert manifest["config"]["image_scoring"] == PROTOCOL
    assert all("/final/" in f["image"] for s in manifest["sequences"] for f in s["frames"])
    other = {"dataset": "kitti", "config": {}, "sequences": []}
    original = copy.deepcopy(other)
    prepare_manifest(other, tmp_path)
    assert other == original


def test_summary_weights_images_by_frame_and_videos_by_sequence(tmp_path):
    (tmp_path / "session.json").write_text(json.dumps({"schema": VERSION}))
    for sequence, count, value in [("short", 1, 0.), ("long", 3, 1.)]:
        path = tmp_path / "benchmark/sintel" / sequence / "baseline/metrics.json"
        path.parent.mkdir(parents=True)
        metrics = {k: value for k in IMAGE_METRICS}
        metrics["absrel_video"] = value
        path.write_text(json.dumps({"scope": "benchmark", "dataset": "sintel", "model": "baseline",
            "sequence": sequence, "metrics": metrics, "metric_counts": {k: count for k in metrics},
            "scoring_protocol": PROTOCOL}))
    row = summarize(tmp_path)[0]
    assert row["absrel_img"] == .75
    assert row["absrel_video"] == .5
    assert row["image_frames"] == 4
    assert row["image_standard"] == PROTOCOL["id"]


def test_obsolete_results_are_not_reported(tmp_path):
    (tmp_path / "session.json").write_text(json.dumps({"schema": "shared-case-v1"}))
    with pytest.raises(ValueError, match="Obsolete"):
        summarize(tmp_path)


def test_registered_da2_paper_settings_reach_loader(monkeypatch):
    from depth_project.models import load_model
    from depth_project.paths import PROJECT
    entry = next(e for e in json.loads((PROJECT/"maintenance/evaluate_models.json").read_text())
                 if e["name"] == "da2_fp32")
    captured = {}
    def adapter(weights, fp32, input_size, backend, **specific):
        captured.update(fp32=fp32, input_size=input_size, **specific)
    monkeypatch.setattr("depth_project.models.adapter_type", lambda name: adapter)
    load_model(entry["adapter"], "weights", **entry["load_param"])
    assert captured == {"fp32": True, "input_size": 518, "align_corners": False}
