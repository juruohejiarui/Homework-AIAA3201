"""Three-model scientific invariants, train-only targets and pair-local feature alignment."""
import copy
from dataclasses import fields

import pytest
import torch

from explore_a.config import FUSIONS, FLOWS, OCCLUSIONS, load_config
from explore_a.geometry.warp import pair_geometry
from explore_a.models.blocks import SmallUNet
from explore_a.models.flow_refine import align_features
from explore_a.models.inputs import FusionInputs
from explore_a.training.losses import fusion_loss
from explore_a.training.prediction import make_fusion, predict, project_to_interval
from explore_a.training.targets import fusion_targets
from test_explore_a_geometry import example


@pytest.fixture
def batch():
    torch.set_num_threads(1)
    data = example(32, 48)
    ramp = torch.linspace(.2, 2., 32*48).reshape(1,1,32,48)
    data['current_depth'] = 2 + ramp
    data['previous_depth'] = 1 + ramp + .05*torch.sin(9*ramp)
    data["gt"] = 1/(.2*data['current_depth']+.2+.02*torch.cos(5*ramp))
    data["gt_valid"] = torch.ones_like(data["gt"], dtype=torch.bool)
    return data


@pytest.mark.parametrize("name", FUSIONS)
def test_gt_free_inputs_finite_backward_fb_cannot_be_overridden(batch, name):
    config = load_config()
    model = make_fusion(name, config)
    assert not any("gt" in item.name for item in fields(FusionInputs))
    output, _ = predict(model, batch, config, "rule")
    if name != "flow_refine":
        assert torch.isfinite(output["depth"]).all() and (output["depth"] >= 0).all()
    else:
        assert float(output["historical_weight"].detach().mean()) == pytest.approx(.05)
        assert not output["correction"].any()
    altered = {**batch, "gt": batch["gt"] * 9, "gt_valid": ~batch["gt_valid"]}
    torch.testing.assert_close(output["depth"], predict(model, altered, config, "rule")[0]["depth"])
    terms = fusion_loss(output, batch, config["training"])
    terms.total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    invalid = {**batch, "forward": torch.ones_like(batch["forward"]) * 10,
               "teacher_visibility": torch.ones_like(batch["current_depth"])}
    absent, _ = predict(model, invalid, config, "mftiq")
    torch.testing.assert_close(absent["depth"], batch["current_depth"], rtol=0, atol=0)
    assert not absent["historical_weight"].any() and not absent["correction"].any()
    empty = {**batch, "gt": torch.full_like(batch["gt"], float("nan")),
             "gt_valid": ~batch["gt_valid"]}
    model.zero_grad()
    loss = fusion_loss(predict(model, empty, config, "rule")[0], empty, config["training"])
    assert loss.total == 0
    loss.total.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_train_targets_known_optimum_small_gaps_and_ambiguous_reset(batch):
    config = load_config()
    current = torch.full_like(batch['gt'], 2.)
    output = {'historical':current+2, 'visibility':torch.ones_like(current)}
    supervised = {**batch, 'current_depth':current, 'gt':current+1}
    targets = fusion_targets(output, supervised, config['training'])
    torch.testing.assert_close(targets.blend, torch.full_like(current, .5))
    assert targets.blend_valid.all() and targets.reset.all()
    close = {**output, 'historical':current+.001}
    assert not fusion_targets(close, supervised, config['training']).blend_valid.any()


@pytest.mark.parametrize("value,target,sign", [(12., 11., -1), (8., 8.5, 1)])
def test_equal_candidates_can_be_repaired_and_projected_without_gt(batch, value, target, sign):
    config = load_config()
    ramp = torch.linspace(0, 1, 48)[None,None,None,:]
    original = torch.full_like(batch["gt"], value) + ramp
    batch.update(current_depth=original,
                 previous_depth=original.clone(),
                 gt=torch.full_like(batch["gt"], target))
    model = make_fusion("flow_refine", config)
    with torch.no_grad():
        model.local_head.bias.fill_(sign * .5)
    output, _ = predict(model, batch, config, "rule")
    assert torch.all((output["depth"] - original) * sign > 0)
    assert output["depth"].min() > 0 and output["correction"].abs().max() <= .1
    projected = project_to_interval(output, {"current_depth": batch["current_depth"]})
    torch.testing.assert_close(projected["depth"], batch["current_depth"])
    fusion_loss(output, batch, config["training"]).total.backward()
    assert model.local_head.bias.grad.abs().sum() > 0


def test_flow_displacement_scaled_to_feature_grid_and_conservative_mask():
    ramp = torch.arange(8.).view(1, 1, 1, 8).expand(1, 2, 4, 8)
    flow = torch.zeros(1, 2, 32, 64)
    flow[:, 0] = -8
    valid = torch.ones(1, 1, 32, 64, dtype=torch.bool)
    warped = align_features(ramp, flow, valid)
    torch.testing.assert_close(warped[..., 1:], ramp[..., :-1])
    valid[..., 8:16] = False
    warped = align_features(ramp, flow, valid)
    assert not warped[..., 1].any()


def test_flow_refine_has_no_cross_pair_state_and_reloads_exactly(batch):
    config = load_config()
    model = make_fusion("flow_refine", config).eval()
    with torch.no_grad():
        model.local_head.weight.normal_(std=.01)
    first = predict(model, batch, config, "rule")[0]["depth"]
    other = {**batch, "current_rgb": torch.zeros_like(batch["current_rgb"])}
    predict(model, other, config, "rule")
    torch.testing.assert_close(first, predict(model, batch, config, "rule")[0]["depth"])
    restored = make_fusion("flow_refine", config).eval()
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(first, predict(restored, batch, config, "rule")[0]["depth"])


def test_matrix_has_four_frontend_runs_and_no_student_training(monkeypatch, tmp_path):
    from explore_a.matrix import run_matrix
    from explore_a.evaluation.branches import matrix_specs, specification
    calls = []
    monkeypatch.setattr("explore_a.matrix.command", lambda config, *args: calls.append(args))
    from explore_a.matrix import experiments
    choices = experiments()
    for config, _ in choices:
        config["paths"]["runs"] = str(tmp_path / "runs")
    monkeypatch.setattr("explore_a.matrix.experiments", lambda: choices)
    run_matrix({}, include_evaluation=False)
    assert len(calls) == 6
    assert all(args[0] == "train" and "student" not in args for args in calls)
    assert len([s for s in matrix_specs() if not s["baseline"]]) == 3
    assert specification("raft", fusion="flow_refine", project_interval=True)["name"].endswith(
        "-projected")
    with pytest.raises(ValueError):
        specification("raft", fusion="unet_guided", project_interval=True)


@pytest.mark.parametrize("name", FUSIONS)
def test_fixed_pair_learning_check(batch, name):
    config = load_config()
    torch.manual_seed(2026)
    model = make_fusion(name, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        terms = fusion_loss(predict(model, batch, config, "rule")[0], batch, config["training"])
        losses.append(float(terms.total.detach()))
        terms.total.backward()
        optimizer.step()
    assert min(losses[-5:]) < losses[0], (name, losses[0], losses[-1])


def test_projection_evaluation_loads_original_checkpoint_and_isolates_results(tmp_path,
                                                                            monkeypatch):
    from test_explore_a_pipeline import online_experiment
    from test_explore_a_online import create_heads
    from explore_a.evaluation.engine import evaluate
    from pathlib import Path
    import numpy as np

    class Depth:
        model = torch.nn.Identity()

        def __init__(self, *args, **kwargs):
            pass

        def predict(self, rgb):
            return np.full(rgb.shape[:2], 2, dtype=np.float32)

    class Flow:
        def __init__(self, *args):
            pass

        def __call__(self, previous, current):
            values = np.zeros((2, *current.shape[:2]), dtype=np.float32)
            return values, values.copy()

    monkeypatch.setattr("explore_a.frontend.make", lambda config: Depth())
    monkeypatch.setattr("explore_a.online.frozen.PairFlow", Flow)
    monkeypatch.setattr("explore_a.online.frozen.check_weights", lambda *args: None)
    config = online_experiment(tmp_path)
    create_heads(config)
    checkpoint = Path(config["paths"]["runs"]) / "farneback-rule-flow_refine/best.pt"
    before = checkpoint.stat().st_mtime_ns
    normal = evaluate(config, "farneback", fusion="flow_refine", device="cpu")
    projected = evaluate(config, "farneback", fusion="flow_refine", device="cpu",
                         project_interval=True)
    assert not normal["projection_ablation"] and projected["projection_ablation"]
    assert projected["experiment"] == normal["experiment"] + "-projected"
    assert projected["fusion_model"]["parameters"] == 357556
    global_only = evaluate(config, "farneback", fusion="flow_refine", device="cpu",
                           global_only=True)
    assert global_only["global_only_ablation"]
    assert global_only["experiment"].endswith("-global-only")
    assert checkpoint.stat().st_mtime_ns == before
    for name in (normal["experiment"], projected["experiment"]):
        assert (Path(config["paths"]["runs"]) / name / "evaluation-test/metrics.json").exists()
