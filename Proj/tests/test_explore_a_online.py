"""Raw streaming, frozen gradients, shared evaluation equivalence and asynchronous failures."""
from pathlib import Path

import numpy as np
import pytest
import torch

from depth_project.io import read_json
from explore_a.config import file_identity
from explore_a.evaluation.branches import matrix_specs
from explore_a.evaluation.engine import evaluate, evaluate_matrix
from explore_a.geometry.warp import pair_geometry
from explore_a.models.occlusion import OcclusionStudent
from explore_a.online.frozen import FrozenRuntime
from explore_a.online.identity import input_identity, teacher_identity
from explore_a.online.loading import read_pair
from explore_a.training.prediction import make_fusion
from test_explore_a_pipeline import online_experiment, fake_backbones as fake_backbones


def test_only_adjacent_raw_depth_reused_and_teacher_conditional(tmp_path, fake_backbones,
                                                              monkeypatch):
    config = online_experiment(tmp_path)
    rows = read_json(config["paths"]["manifest"])["samples"]
    depth, _, _ = fake_backbones
    required = []
    monkeypatch.setattr("explore_a.online.frozen.check_weights",
                        lambda config, names: required.append(names))
    runtime = FrozenRuntime(config, ["farneback"], False, "cpu")
    assert required == [["depthpro"]] and runtime.teacher is None
    for row in rows[:2]:
        runtime.compute(read_pair(row, config["data"]), "farneback", ordered=True)
    assert depth.calls == 3
    runtime.compute(read_pair(rows[2], config["data"]), "farneback", ordered=True)
    assert depth.calls == 5  # New sequence: no cross-sequence reuse.
    for _ in range(2):
        runtime.compute(read_pair(rows[0], config["data"]), "farneback", ordered=False)
    assert depth.calls == 9
    assert runtime.previous_key is None
    other = FrozenRuntime(config, ["raft"], True, "cpu")
    assert required[-1] == ["depthpro", "raft", "mftiq", "resnet", "dinov2"]
    assert other.teacher is not None
    assert not list(tmp_path.rglob("*.npz"))


def test_frozen_outputs_are_safe_for_head_backward(tmp_path, monkeypatch):
    config = online_experiment(tmp_path)

    class Depth:
        def __init__(self, *_, **__):
            self.model = torch.nn.Conv2d(1, 1, 1)
            torch.nn.init.zeros_(self.model.weight)
            torch.nn.init.constant_(self.model.bias, 2)

        def predict(self, rgb):
            value = self.model(torch.ones(1, 1, *rgb.shape[:2]))
            assert not value.requires_grad
            return value[0, 0].numpy()

    monkeypatch.setattr("explore_a.online.frozen.DepthProAdapter", Depth)
    runtime = FrozenRuntime(config, ["farneback"], False, "cpu")
    row = read_json(config["paths"]["manifest"])["samples"][0]
    sample = runtime.compute(read_pair(row, config["data"]), "farneback")
    data = {key: value.unsqueeze(0) for key, value in sample.items()}
    assert not any(value.is_inference() for value in data.values())
    model = make_fusion("residual", config)
    geometry = pair_geometry(data, config["geometry"])
    model(data, geometry, geometry["rule"].float())["depth"].mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(not p.requires_grad and p.grad is None for p in runtime.depth.model.parameters())


def create_heads(config):
    root = Path(config["paths"]["runs"])
    for flow in ("raft", "farneback"):
        path = root / f"occlusion-{flow}/best.pt"
        path.parent.mkdir(parents=True)
        torch.save({"model": OcclusionStudent().state_dict(), "identity": {
            "input": input_identity(config, flow), "teacher": teacher_identity(config),
            "experiment": {"flow": flow, "occlusion": "distillation", "fusion": "student"},
        }}, path)
    for spec in matrix_specs():
        if spec["baseline"]:
            continue
        identity = {"input": input_identity(config, spec["flow"]),
                    "experiment": {key: spec[key] for key in ("flow", "occlusion", "fusion")}}
        if spec["occlusion"] == "mftiq":
            identity["teacher"] = teacher_identity(config)
        if spec["occlusion"] == "student":
            identity["student"] = file_identity(root / f"occlusion-{spec['flow']}/best.pt")
        path = root / spec["name"] / "best.pt"
        path.parent.mkdir(parents=True)
        torch.save({"model": make_fusion(spec["fusion"], config).state_dict(),
                    "identity": identity}, path)


def test_all_23_shared_branches_match_independent_and_share_depth(tmp_path, fake_backbones):
    config = online_experiment(tmp_path)
    create_heads(config)
    depth, _, _ = fake_backbones
    results = evaluate_matrix(config, device="cpu")
    assert len(results) == 23 and depth.calls == 3
    for spec, shared in zip(matrix_specs(), results):
        separate = evaluate(config, spec["flow"], spec["occlusion"], spec["fusion"],
                            spec["baseline"], device="cpu")
        assert shared["latency"]["warm_ms"] is None
        assert shared["latency"]["fusion_warm_ms"] > 0
        assert separate["latency"]["warm_ms"] > 0
        for metric, value in shared["macro_mean"].items():
            np.testing.assert_allclose(value, separate["macro_mean"][metric], rtol=1e-6, atol=1e-7)
    assert not list(tmp_path.rglob("*.npz"))


@pytest.mark.parametrize("shared", [False, True])
def test_async_panel_failure_prevents_success_summary(tmp_path, monkeypatch, shared):
    config = online_experiment(tmp_path)
    config["evaluation"]["visualizations_per_sequence"] = 1
    if shared:
        create_heads(config)
        marker = Path(config["paths"]["runs"]) / "matrix-test.json"
        marker.write_text('{"old": true}')

    def fail(*_):
        raise OSError("disk failure")

    monkeypatch.setattr("explore_a.evaluation.output.save_panel", fail)
    with pytest.raises(OSError, match="disk failure"):
        if shared:
            evaluate_matrix(config, device="cpu")
        else:
            evaluate(config, "farneback", baseline="raw", device="cpu")
    root = Path(config["paths"]["runs"])
    assert not list(root.glob("*/evaluation-test/metrics.json"))
    assert not (root / "matrix-test.json").exists()


def test_legacy_identity_is_explicitly_rejected(tmp_path):
    from explore_a.config import bind_identity
    from depth_project.io import write_json
    path = tmp_path / "identity.json"
    write_json(path, {"experiment": "previous protocol"})
    with pytest.raises(ValueError, match="NEW runs directory"):
        bind_identity(path, {"input": {"schema": 2}})


def test_raft_bidirectional_batch_order_and_oom_fallback():
    from explore_a.models.flow import PairFlow
    backend = PairFlow.__new__(PairFlow)
    backend.name, backend.device, backend.updates = "raft", "cpu", 12
    backend.bidirectional_batch = True

    def model(first, second, num_flow_updates):
        assert num_flow_updates == 12
        assert first.dtype == second.dtype == torch.float32
        return [(second - first)[:, :2]]

    backend.model = model
    previous = np.zeros((8, 8, 3), np.uint8)
    current = np.ones_like(previous) * 100
    batch = backend(previous, current)
    single = (backend.one_way(previous, current), backend.one_way(current, previous))
    for first, second in zip(batch, single):
        np.testing.assert_allclose(first, second, rtol=1e-6)
    assert (batch[0] > 0).all() and (batch[1] < 0).all()

    def oom(*_):
        raise torch.cuda.OutOfMemoryError()

    backend.bidirectional = oom
    result = backend(previous, current)
    assert not backend.bidirectional_batch
    np.testing.assert_array_equal(result[0], single[0])
