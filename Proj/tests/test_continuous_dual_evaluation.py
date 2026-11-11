"""Continuous cohorts and gauge diagnostics, without foundation models or datasets."""
from collections import Counter, defaultdict

import numpy as np

from explore_a.evaluation.alignment import RelativeScores, ordinal
from explore_a.evaluation.oracle_scoring import fixed_metrics, fields, mapping
from explore_a.training.dev_subset import select_dev, selection_identity


def test_continuous_selection_source_ratio_and_gap_isolation():
    rows = []
    for source in ("kitti", "nyuv2", "tum"):
        for video in range(3):
            for i in range(100):
                # Two disconnected chains, not an artificial bridge over the gap.
                before = i if i < 50 else i+100
                rows.append({"dataset": source, "sequence": str(video), "camera": "rgb",
                             "id": f"{source}:{video}:{i}", "previous_id": before,
                             "current_id": before+1})
    selected = select_dev(rows, .4)
    assert Counter(r["dataset"] for r in selected) == {"kitti": 135, "nyuv2": 135, "tum": 90}
    assert selected == select_dev(rows[::-1], .4)
    assert selection_identity(selected) == selection_identity(select_dev(rows, .4))
    groups = defaultdict(list)
    for row in selected:
        groups[row["evaluation_clip"]].append(row)
    for group in groups.values():
        for a, b in zip(group, group[1:]):
            assert a["current_id"] == b["previous_id"]
    assert all(len({r["sequence"] for r in selected if r["dataset"] == s}) == 3
               for s in ("kitti", "nyuv2", "tum"))


def test_image_alignment_hides_drift_video_does_not():
    target = np.linspace(1, 8, 256, dtype=np.float32).reshape(16, 16)
    scorer = RelativeScores()
    image = []
    for i, (scale, shift) in enumerate(((1., 0.), (3., 2.))):
        row = {"id": str(i), "dataset": "nyuv2", "sequence": "s", "camera": "rgb"}
        image.append(scorer.add(row, scale/target+shift, target, target > 0, (.1, 10.)))
    video = list(scorer.finish())
    assert max(v["absrel_img"] for v in image) < 1e-5
    assert max(v[1]["absrel_video"] for v in video) > .1
    assert all(v["ordinal_accuracy"] == 1 for v in image)


def test_ordinal_no_pairs_and_ties():
    target = np.array([1., 2., 4.])
    prediction = 1/target
    assert ordinal(prediction, target, "fixed") == ordinal(7*prediction+3, target, "fixed")
    assert ordinal(np.ones(3), target, "fixed")["ordinal_accuracy"] == 0
    assert ordinal(prediction, np.ones(3), "fixed") == {}
    assert ordinal(np.array([]), np.array([]), "fixed") == {}


def test_oracle_common_masks_nested_bounds_and_fallback():
    gt = np.linspace(1, 8, 256, dtype=np.float32).reshape(16, 16)
    raw = 1/gt + .08*np.sin(gt*2)
    history = raw+.03
    valid = np.ones_like(gt, bool)
    common = valid.copy()
    common[:, :4] = False
    valid[0] = False
    packed = np.stack((raw, history, history, gt, valid, common, np.ones_like(gt)))
    gauge = mapping(raw, gt, valid)
    predictions, _ = fields(packed, gauge)
    names = ("oracle_additive10", "oracle_aligned", "raw")
    losses = [fixed_metrics(predictions[n], gt, common & valid, gauge, (.1, 10.))["inverse_loss"]
              for n in names]
    assert losses[0] <= losses[1] <= losses[2]
    for value in predictions.values():
        np.testing.assert_array_equal(value[~(common & valid)], raw[~(common & valid)])
    assert fixed_metrics(raw, gt, np.zeros_like(common), gauge, (.1, 10.)) == {}


def test_report_exposes_both_gauges_and_ordinal(tmp_path):
    from depth_project.io import write_json
    from explore_a.evaluation.comparison_report import report
    write_json(tmp_path/"dev/fusion/head/evaluation-dev/metrics.json", {
        "experiment": "head", "frontend": {"name": "da2"}, "by_dataset": {"nyuv2": {
            "absrel_img": .1, "rmse_img": .2, "delta1_img": .9,
            "absrel_video": .3, "rmse_video": .4, "delta1_video": .8,
            "ordinal_accuracy": .95}}})
    rows = report(tmp_path)
    assert {r["scope"] for r in rows} == {"dev-img", "dev-video"}
    assert {r["abs_rel"] for r in rows} == {.1, .3}
    assert all(r["ordinal_accuracy"] == .95 for r in rows)
