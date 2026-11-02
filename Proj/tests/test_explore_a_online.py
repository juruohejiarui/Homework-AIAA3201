"""Raw streaming, frozen gradients, shared evaluation equivalence and asynchronous failures."""
from pathlib import Path

import numpy as np
import pytest
import torch

from depth_project.io import read_json
from explore_a.evaluation.branches import matrix_specs
from explore_a.evaluation.engine import evaluate, evaluate_matrix
from explore_a.geometry.warp import pair_geometry
from explore_a.online.factory import make_runtime
from explore_a.online.identity import input_identity, teacher_identity
from explore_a.online.loading import read_pair
from explore_a.training.prediction import make_fusion, predict
from test_explore_a_pipeline import online_experiment, fake_backbones as fake_backbones


def create_heads(config):
    root = Path(config["paths"]["runs"])
    from explore_a.evaluation.branches import specification
    specs = matrix_specs()
    for spec in specs:
        if spec["baseline"]:
            continue
        identity = {"frontend":{"name":"da2", "kind":"relative_inverse"},
                    "input": input_identity(config, spec["flow"]),
                    "experiment": {key: spec[key] for key in ("flow", "occlusion", "fusion")}}
        if spec["occlusion"] == "mftiq":
            identity["teacher"] = teacher_identity(config)
        path = root / spec["name"] / "best.pt"
        path.parent.mkdir(parents=True)
        torch.save({"model": make_fusion(spec["fusion"], config).state_dict(),
                    "identity": identity}, path)


def test_shared_candidates_match_independent_and_share_depth(tmp_path, fake_backbones):
    config = online_experiment(tmp_path)
    create_heads(config)
    depth, _, _ = fake_backbones
    results = evaluate_matrix(config, device="cpu")
    assert len(results) == 4 and depth.calls == 3
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
