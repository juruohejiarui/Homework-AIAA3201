"""Decomposition semantics, correction gradients and diagnostic-only GradNorm targets."""
from dataclasses import fields

import pytest
import torch

from explore_a.config import load_config
from explore_a.models.correction import bounded_correction, CorrectionContext
from explore_a.models.inputs import FusionInputs
from explore_a.training.checkpoint import load_checkpoint
from explore_a.training.correction import correction_losses, correction_targets
from explore_a.training.gradnorm import candidate_weights, measure
from explore_a.training.prediction import make_fusion, predict, global_only
from explore_a.observability.scalars import scalars
from test_explore_a_geometry import example


def test_bounded_range_opposing_local_and_zero_initialization_derivative():
    global_value = torch.linspace(-.1, .1, 31).reshape(-1, 1)
    q = torch.linspace(-1, 1, 51).reshape(1, -1)
    result = bounded_correction(global_value, q, .1)
    assert result.min() >= -.10000001 and result.max() <= .10000001
    torch.testing.assert_close(result[:, 0], torch.full((31,), -.1))
    torch.testing.assert_close(result[:, -1], torch.full((31,), .1))
    assert bounded_correction(torch.tensor(.099), torch.tensor(-.8), .1) < 0
    q0 = torch.tensor(0., requires_grad=True)
    bounded_correction(torch.tensor(0.), q0, .1).backward()
    assert q0.grad == pytest.approx(.1)


@pytest.mark.parametrize("truth", [[1.8, 2.2], [2.5, 2.5], [1.5, 1.5]])
def test_targets_reconstruct_bounded_oracle_and_are_detached(truth):
    mixed = torch.full((1, 1, 1, 2), 2., requires_grad=True)
    output = {"mixed_depth": mixed, "visibility": torch.ones_like(mixed)}
    data = {"gt": torch.tensor(truth).reshape_as(mixed),
            "gt_valid": torch.ones_like(mixed, dtype=torch.bool)}
    g, q, valid = correction_targets(output, data, .1)
    expected = (data["gt"] / 2 - 1).clamp(-.1, .1)
    torch.testing.assert_close(bounded_correction(g, q, .1), expected)
    assert not g.requires_grad and not q.requires_grad and valid.all()
    torch.testing.assert_close(g, expected.mean((-2, -1), keepdim=True))


def test_coverage_pooling_film_identity_and_empty_history():
    block = CorrectionContext()
    memory = torch.randn(2, 64, 2, 3)
    decoded = torch.randn(2, 16, 16, 24)
    valid = torch.zeros(2, 1, 16, 24, dtype=torch.bool)
    valid[0, :, :8, :8] = True
    captured = []
    hook = block.context[0].register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    g, local = block(memory, decoded, valid, .1)
    hook.remove()
    torch.testing.assert_close(captured[0][0], memory[0, :, 0, 0])
    assert not captured[0][1].any() and not g.any()
    torch.testing.assert_close(local, decoded, rtol=0, atol=0)


def test_new_losses_empty_gt_zero_range_and_per_frame_reduction():
    config = load_config()
    depth = torch.ones(2, 1, 2, 2, requires_grad=True)
    output = {"depth": depth, "mixed_depth": depth, "visibility": torch.ones_like(depth),
              "global_correction": torch.zeros(2, 1, 1, 1, requires_grad=True),
              "local_control": torch.zeros_like(depth, requires_grad=True)}
    mask = torch.ones_like(depth, dtype=torch.bool)
    mask[0, :, :, 1] = False
    truth = torch.cat([torch.ones(1, 1, 2, 2) * 1.05, torch.ones(1, 1, 2, 2) * .95])
    data = {"gt": truth, "gt_valid": mask}
    g, q = correction_losses(output, data, config["training"])
    assert float(g.detach()) == pytest.approx(.45) and q == 0
    empty = {"gt": truth * float("nan"), "gt_valid": ~torch.ones_like(mask)}
    losses = correction_losses(output, empty, config["training"])
    sum(losses).backward()
    assert all(value == 0 for value in losses)
    assert torch.isfinite(output["global_correction"].grad).all()
    zero = {**config["training"], "correction_fraction": 0}
    assert all(value == 0 for value in correction_losses(output, data, zero))


def test_model_gt_independence_global_ablation_and_old_checkpoint_message(tmp_path):
    torch.set_num_threads(1)
    config = load_config()
    model = make_fusion("flow_refine", config)
    data = example(32, 48)
    data.update(gt=torch.ones_like(data["current_depth"]) * 3,
                gt_valid=torch.ones_like(data["current_depth"], dtype=torch.bool))
    assert all("gt" not in field.name for field in fields(FusionInputs))
    with torch.no_grad():
        model.correction_context.global_head.bias.fill_(.3)
        model.local_head.bias.fill_(-.5)
    output = predict(model, data, config, "rule")[0]
    changed = predict(model, {**data, "gt": data["gt"] * 9}, config, "rule")[0]
    torch.testing.assert_close(output["depth"], changed["depth"], rtol=0, atol=0)
    only = global_only(output, data)
    torch.testing.assert_close(only["depth"], (output["mixed_depth"] + output["field_scale"]
                                * output["global_correction"]).clamp_min(0))
    assert not only["local_correction"].any()
    path = tmp_path / "old.pt"
    torch.save({"model": {"global_head.2.weight": torch.zeros(32, 64)}}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_checkpoint(path, model)


def test_gradnorm_candidates_and_measurements_do_not_apply_gradients():
    torch.set_num_threads(1)
    weights = candidate_weights({"a": 1., "b": 1.}, {"a": 1., "b": 1.},
                                {"a": 2., "b": 1.})
    assert weights == pytest.approx({"a": 2 / 3, "b": 4 / 3})
    assert candidate_weights({"a": 1.}, {"a": 1.}, {"a": 0.}) is None
    config = load_config()
    model = make_fusion("flow_refine", config)
    data = example(32, 48)
    data.update(gt=torch.ones_like(data["current_depth"]) * 3,
                gt_valid=torch.ones_like(data["current_depth"], dtype=torch.bool))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = measure(model, {s: data for s in ("kitti", "tum", "nyuv2")}, config)
    assert set(result["objectives"]) == {"depth", "total"}
    assert all(p.grad is None for p in model.parameters())
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    tags = dict(scalars("train", {"step": 1, "correction_metrics": {
        "tum": {"global_mean": .02}}, "losses_by_dataset": {"tum": {"aux_depth_ratio": 2}}}))
    assert tags["train/tum/correction/global_mean"] == .02
    assert tags["train/tum/aux_depth_ratio"] == 2
    from explore_a.observability.table import tagged
    assert tagged(1, "train/tum/correction/global_mean", .02)["dataset"] == "tum"


def test_diagnostic_isolated_output_and_short_cached_loop(tmp_path, monkeypatch):
    from explore_a.training import diagnostic
    from depth_project.io import read_json
    torch.set_num_threads(1)
    config = load_config()
    data = example(32, 48)
    data.update(gt=torch.ones_like(data["current_depth"]) * 3,
                gt_valid=torch.ones_like(data["current_depth"], dtype=torch.bool))
    item = {key: value[0] for key, value in data.items()}
    pairs = {source: [({"dataset": source, "id": source}, item)]
             for source in ("kitti", "tum", "nyuv2")}
    monkeypatch.setattr(diagnostic, "load_probe", lambda config: (pairs, [], {}))
    root = tmp_path / "diagnostic"
    result = diagnostic.run(config, root, "cpu", steps=5)
    assert result["steps"] == 5 and not result["weights_applied"]
    assert (root / "head.pt").exists()
    assert "ordinary_backward" in read_json(root / "comparison.json")
    with pytest.raises(ValueError, match="empty diagnostic"):
        diagnostic.run(config, root, "cpu", steps=5)
