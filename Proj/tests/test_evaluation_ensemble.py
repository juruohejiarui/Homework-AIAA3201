"""Pixel means, dependency resolution, shared execution and immutable ensemble resume."""
from concurrent.futures import Future
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from explore_a.evaluation.ensemble import EnsembleFields, mean_fields, plan
from explore_a.evaluation import session
from explore_a.training.prediction import make_fusion
from test_explore_a_pipeline import fake_backbones, online_experiment


def test_arbitrary_member_count_and_nested_resolution():
    leaves = [{"name": str(i)} for i in range(17)]
    first = {"name": "all", "kind": "ensemble", "members": [e["name"] for e in leaves]}
    second = {"name": "nested", "kind": "ensemble", "members": ["all", "0"]}
    models, ensembles, requested = plan([*leaves, first, second], ["nested"])
    assert models == leaves
    assert ensembles == [first, second]
    assert requested == ["nested"]
    fields = [np.full((3, 4), i, np.float32) for i in range(17)]
    np.testing.assert_array_equal(mean_fields(fields), np.full((3, 4), 8, np.float32))
    np.testing.assert_array_equal(mean_fields(fields[:1]), fields[0])


@pytest.mark.parametrize("members", [[], ["a", "a"], "a", [3], ["missing"], ["avg"]])
def test_invalid_ensemble_members_fail_early(members):
    with pytest.raises(ValueError):
        plan([{"name": "a"}, {"name": "avg", "kind": "ensemble", "members": members}],
             ["avg"])


@pytest.mark.parametrize("deferred", [False, True])
def test_streamed_and_sequence_fields_average_before_clipping(tmp_path, deferred):
    definition = {"name": "avg", "kind": "ensemble", "members": ["a", "b"]}
    fields = EnsembleFields(tmp_path, [definition], deferred)
    for index in range(2):
        fields.store("a", index, np.full((3, 4), -2 + index, np.float32),
                     "relative_inverse", (3, 4))
        fields.store("b", index, np.full((3, 4), 4 + index, np.float32),
                     "relative_inverse", (3, 4))
        if not deferred:
            value, representation = fields.predictions(index, ["avg"])["avg"]
            np.testing.assert_array_equal(value, np.full((3, 4), 1 + index, np.float32))
            assert representation == "relative_inverse"
            assert not fields.current
    if deferred:
        for index in range(2):
            value, _ = fields.predictions(index, ["avg"])["avg"]
            np.testing.assert_array_equal(value, np.full((3, 4), 1 + index, np.float32))
    else:
        assert not list(tmp_path.rglob("*.npy"))


def test_incompatible_units_and_invalid_fields_are_rejected(tmp_path):
    fields = EnsembleFields(tmp_path, [{"name": "avg", "kind": "ensemble",
                                        "members": ["a", "b"]}])
    fields.store("a", 0, np.ones((3, 4)), "relative_inverse", (3, 4))
    fields.store("b", 0, np.ones((3, 4)), "metric_m", (3, 4))
    with pytest.raises(ValueError, match="representations"):
        fields.predictions(0, ["avg"])
    with pytest.raises(ValueError, match="matching HW"):
        mean_fields([np.ones((2, 3)), np.ones((3, 2))])
    with pytest.raises(ValueError, match="Nonfinite"):
        mean_fields([np.full((2, 3), np.nan)])


def registry_fixture(tmp_path):
    config = online_experiment(tmp_path)
    config["training"]["source_weights"] = {name: 1 for name in config["data"]["datasets"]}
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config))
    entries = []
    for index, name in enumerate(("a", "b", "c"), 1):
        model = make_fusion("unet_guided", config)
        with torch.no_grad():
            next(model.parameters()).flatten()[0] = index
        path = tmp_path / f"{name}.pt"
        torch.save({"model": model.state_dict(), "identity": {
            "training": config["training"], "frontend": {"name": "da2"},
            "experiment": {"flow": "farneback", "occlusion": "rule", "fusion": "unet_guided"}},
                    "step": 1, "best": .1}, path)
        entries.append({"name": name, "config": str(cfg), "checkpoint": str(path)})
    entries.append({"name": "avg", "kind": "ensemble", "members": ["a", "b", "c"]})
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(entries))
    return config, registry, entries


def test_ensemble_only_shares_members_resumes_and_checks_membership(tmp_path, monkeypatch,
                                                                  fake_backbones):
    # This test mutates member declarations, independently of edits in the live workspace.
    monkeypatch.setattr("depth_project.models.provenance.shared_signature", lambda: "test-source")
    config, registry, entries = registry_fixture(tmp_path)
    calls, averages = [], []

    def predict(model, data, config, occlusion, geometry):
        assert not model.training
        assert all(not p.requires_grad for p in model.parameters())
        gain = float(next(model.parameters()).flatten()[0])
        calls.append(gain)
        grid = torch.linspace(0, 1, data["current_depth"].shape[-1])
        return {"depth": data["current_depth"] + gain * grid.square()}, geometry

    def average(fields):
        result = mean_fields(fields)
        expected = np.mean(np.stack(fields).astype(np.float64), axis=0).astype(np.float32)
        np.testing.assert_array_equal(result, expected)
        averages.append(result)
        return result

    monkeypatch.setattr(session, "predict", predict)
    monkeypatch.setattr("explore_a.evaluation.ensemble.mean_fields", average)
    output = tmp_path / "evaluation"
    result = session.run(registry, output, ["test"], tmp_path, tmp_path, ["avg"], device="cpu")
    assert {row["model"] for row in result} == {"avg", "raw_canvas"}
    assert calls == [1., 2., 3.] * 2
    assert len(averages) == 2
    assert len(list(output.glob("models/*/checkpoint.pt"))) == 3
    before = fake_backbones[0].calls
    assert session.run(registry, output, ["test"], tmp_path, tmp_path,
                       ["avg"], device="cpu") == result
    assert len(calls) == 6 and fake_backbones[0].calls == before
    baseline = next(output.glob("test/*/*/raw_canvas/metrics.json"))
    stamp = baseline.stat().st_mtime_ns
    entries.append({"name": "avg2", "kind": "ensemble", "members": ["a", "b"]})
    registry.write_text(json.dumps(entries))
    expanded = session.run(registry, output, ["test"], tmp_path, tmp_path,
                            ["avg2"], device="cpu")
    assert {row["model"] for row in expanded} == {"avg", "avg2", "raw_canvas"}
    assert baseline.stat().st_mtime_ns == stamp
    assert calls[6:] == [1., 2.] * 2
    entries.pop()
    entries[-1]["members"] = ["a", "b"]
    registry.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match="Previously evaluated model changed"):
        session.run(registry, output, ["test"], tmp_path, tmp_path, ["avg"], device="cpu")


@pytest.mark.parametrize("capability", ["image", "sequence"])
def test_frozen_reference_members_share_the_ensemble_path(tmp_path, monkeypatch,
                                                         fake_backbones, capability):
    config, registry, entries = registry_fixture(tmp_path)
    entries.insert(-1, {"name": "reference", "kind": "frozen", "adapter": "vda",
                       "capability": capability, "config": entries[0]["config"],
                       "weights": config["weights"]["da2"]})
    entries[-1]["members"] = ["a", "reference"]
    registry.write_text(json.dumps(entries))
    averages = []

    class Reference:
        def forward(self, request, **parameters):
            from depth_project.models.api import ModelOutput
            predictions = [torch.full(rgb.shape[:2], 4. + 4 * index)
                           for index, rgb in enumerate(request.rgb)]
            return ModelOutput(torch.stack(predictions), "relative_inverse")

    def average(fields):
        result = mean_fields(fields)
        averages.append(result)
        return result

    monkeypatch.setattr(session, "reference", lambda binding: Reference())
    monkeypatch.setattr("explore_a.evaluation.ensemble.mean_fields", average)
    result = session.run(registry, tmp_path / "evaluation", ["test"], tmp_path, tmp_path,
                          ["avg"], device="cpu")
    assert {row["model"] for row in result} == {"avg", "raw_canvas"}
    assert len(averages) == 2
    for i, prediction in enumerate(averages):
        expected = 3. + (2 * i if capability == "sequence" else 0)
        np.testing.assert_array_equal(prediction, np.full((16, 24), expected, np.float32))


def test_remote_member_fields_are_averaged_without_extra_rpc(tmp_path, monkeypatch,
                                                            fake_backbones):
    config, registry, _ = registry_fixture(tmp_path)
    leaves, ensembles, requested = plan(session.entries(registry), ["avg"])
    bindings = [session.bind(e, tmp_path / "snapshots", "cpu") for e in leaves]
    case = next(session.cohorts("test", config, tmp_path, tmp_path))
    monkeypatch.setattr("explore_a.evaluation.transport.case_route", lambda *args: "dgx")
    calls = []

    class Remote:
        provenance = {"test": "member fields from remote protocol"}

        def submit(self, raw, native, history, names):
            calls.append(names)
            field = np.ones(raw["canvas"], np.float32)
            future = Future()
            future.set_result({"raw_canvas": field,
                               **{name: field * i for i, name in enumerate(names, 1)}})
            return future

    class Runtime:
        def statistics(self):
            return {}

    results = session.evaluate_case(case, bindings, tmp_path / "evaluation", Runtime(),
                                    "cpu", Remote(), ensembles, requested)
    assert {row["model"] for row in results} == {"avg", "raw_canvas"}
    assert calls == [["a", "b", "c"]] * 2


def test_ensemble_uses_v2_sintel_ac_after_raw_field_average(tmp_path, monkeypatch, fake_backbones):
    from explore_a.evaluation.sintel_ac import PROTOCOL, image_scores
    from explore_a.data.prepare import load_manifest
    from explore_a.online.loading import read_pair
    config, registry, entries = registry_fixture(tmp_path)
    entries[-1]["members"] = ["a", "b"]
    leaves, ensembles, requested = plan(entries, ["avg"])
    bindings = [session.bind(e, tmp_path / "snapshots", "cpu") for e in leaves]
    raw = read_pair(load_manifest(config)["samples"][-1], config["data"])
    gt = np.linspace(1, 150, 16 * 24, dtype=np.float32).reshape(16, 24)
    inverse = 1 / np.minimum(gt, 72)
    perturbation = .1 * np.sin(np.arange(gt.size).reshape(gt.shape)).astype(np.float32)
    first, second = -inverse + perturbation, 3 * inverse - perturbation
    expected = image_scores(mean_fields([first, second]), gt)
    assert first.min() < 0
    case = {"scope": "benchmark", "dataset": "sintel", "sequence": "scene", "camera": "video",
            "key": "v2", "frames": [], "protocol": {"max_depth": 70, "image_scoring": PROTOCOL}}
    monkeypatch.setattr(session, "samples", lambda *args: iter([(raw, gt, gt < 70, True)]))
    monkeypatch.setattr("explore_a.evaluation.transport.case_route", lambda *args: "dgx")

    class Remote:
        def submit(self, *args):
            result = Future()
            result.set_result({"raw_canvas": inverse, "raw_native": inverse,
                               "a": first, "b": second})
            return result

    class Runtime:
        def statistics(self):
            return {}

    records = session.evaluate_case(case, bindings, tmp_path / "evaluation", Runtime(),
                                    "cpu", Remote(), ensembles, requested)
    measured = next(row for row in records if row["model"] == "avg")
    assert measured["scoring_protocol"] == PROTOCOL
    assert measured["ensemble"]["members"] == ["a", "b"]
    for key, value in expected.items():
        assert measured["metrics"][key] == pytest.approx(value, abs=1e-8)
