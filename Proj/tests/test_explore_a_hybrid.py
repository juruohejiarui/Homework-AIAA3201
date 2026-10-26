"""Synthetic per-component fallback, lazy model loading, RNG isolation and frozen outputs."""
from pathlib import Path
import json
import random

import numpy as np
import pytest
import torch

from depth_project.io import read_json
from explore_a.cache.hybrid import HybridRuntime
from explore_a.cache.paths import artifact
from explore_a.cache.producer import Producer
from explore_a.online.loading import read_pair
from explore_a.training.prediction import make_fusion, predict
from test_explore_a_distributed import prepared as prepared
from test_explore_a_pipeline import fake_backbones as fake_backbones


def setup_hybrid(prepared):
    config, base, root, name = prepared
    job = read_json(root / "jobs" / name / "job.json")
    Producer(config, "cpu").generate(root, job, sources=read_json(
        root / "jobs" / name / "sources.json"), host="local", minimum_gib=0)
    config["execution"] = {"backend": "hybrid"}
    config["cache"] = {"root": str(base), "recipe": root.name}
    rows = read_json(config["paths"]["manifest"])["samples"]
    row = next(row for row in rows if row["id"] == job["rows"][0]["id"])
    return config, root, row, rows


@pytest.mark.parametrize("missing", range(8))
def test_component_combinations_load_only_needed_models(prepared, missing):
    config, root, row, _ = setup_hybrid(prepared)
    from explore_a.online.identity import input_identity
    identity = input_identity(config, "raft")
    for bit, kind in ((1, "frames"), (2, "flow/raft"), (4, "teacher/raft")):
        if missing & bit:
            artifact(root, kind, row).unlink()
    before = set(root.rglob("*.npz"))
    runtime = HybridRuntime(config, ["raft"], True, "cpu")
    data = runtime.compute(read_pair(row, config["data"]), "raft", True)
    expected = {name for bit, name in ((1, "depth"), (2, "raft"), (4, "teacher")) if missing & bit}
    assert set(runtime.frozen.models) == expected
    stats = runtime.statistics()
    assert stats.get("depth_online", 0) == bool(missing & 1)
    assert stats.get("flow_raft_online", 0) == bool(missing & 2)
    assert stats.get("teacher_raft_online", 0) == bool(missing & 4)
    assert stats["depth_hit"] == (1 if missing & 1 else 2)
    assert set(root.rglob("*.npz")) == before
    assert input_identity(config, "raft") == identity
    assert all(not value.is_inference() for value in data.values())
    model = make_fusion("unet", config)
    output, _ = predict(model, {key: value.unsqueeze(0) for key, value in data.items()},
                        config, "rule")
    output["depth"].mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    for frontend in runtime.frozen.models.values():
        if hasattr(frontend, "model"):
            assert all(not parameter.requires_grad and parameter.grad is None
                       for parameter in frontend.model.parameters())


def test_bad_cache_falls_back_without_overwriting(prepared):
    config, root, row, _ = setup_hybrid(prepared)
    bad = artifact(root, "frames", row)
    bad.write_bytes(b"power loss")
    runtime = HybridRuntime(config, ["farneback"], False, "cpu")
    data = runtime.compute(read_pair(row, config["data"]), "farneback")
    assert runtime.statistics()["depth_invalid"] == 1
    assert runtime.statistics()["depth_online"] == 1
    assert set(runtime.frozen.models) == {"depth"}
    assert bad.read_bytes() == b"power loss"
    expected = torch.full_like(data["current_depth"], 2)
    torch.testing.assert_close(data["current_depth"], expected)


def test_lazy_initialization_preserves_rng_and_sequence_state(prepared, monkeypatch,
                                                            fake_backbones):
    config, root, row, rows = setup_hybrid(prepared)
    for path in list(root.rglob("*.npz")):
        path.unlink()
    original = fake_backbones[0]
    class RandomDepth(original):
        def __init__(self, *args, **kwargs):
            random.random()
            np.random.rand()
            torch.rand(2)
            super().__init__(*args, **kwargs)
    monkeypatch.setattr("explore_a.online.frozen.DepthProAdapter", RandomDepth)
    runtime = HybridRuntime(config, ["raft"], True, "cpu")
    python, numpy, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    raw = read_pair(row, config["data"])
    runtime.compute(raw, "raft", True, ordered=True)
    assert random.getstate() == python
    assert np.array_equal(np.random.get_state()[1], numpy[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)
    next_row = next(r for r in rows if r["sequence"] == row["sequence"]
                    and r["previous_id"] == row["current_id"])
    runtime.compute(read_pair(next_row, config["data"]), "raft", ordered=True)
    assert runtime.statistics()["depth_reuse"] == 1
    other = next(r for r in rows if r["sequence"] != row["sequence"])
    runtime.compute(read_pair(other, config["data"]), "raft", ordered=True)
    assert runtime.statistics()["depth_reuse"] == 1


def test_hybrid_training_dev_and_shared_evaluation(prepared):
    config, root, _, _ = setup_hybrid(prepared)
    from explore_a.training.engine import run_training
    from explore_a.evaluation.engine import evaluate_specs
    from explore_a.evaluation.branches import specification
    config["training"]["heldout_every"] = 0
    run_training(config, "raft", "rule", "unet", device="cpu")
    run = Path(config["paths"]["runs"]) / "raft-rule-unet"
    training = [json.loads(line) for line in (run / "training.jsonl").read_text().splitlines()]
    assert training[-1]["input_stats"]
    assert "train_inputs/" in (run / "scalars.csv").read_text()
    specs = [specification("raft", baseline="raw"),
             specification("farneback", baseline="rule-fixed")]
    combined = evaluate_specs(config, specs, device="cpu")
    single = evaluate_specs(config, specs[:1], device="cpu")[0]
    assert combined[0]["by_dataset"] == single["by_dataset"]
    assert single["latency"]["mode"] == "independent_hybrid"
    assert single["latency"]["input_stats"]["depth_online"] == 3
    assert combined[0]["latency"]["mode"] == "shared_hybrid"
