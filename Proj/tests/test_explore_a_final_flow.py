"""Two-frame input limits, anchored increments, motion-aware targets and shared evaluation."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from explore_a.config import FUSIONS, load_config
from explore_a.geometry.warp import pair_geometry, warp
from explore_a.models.inputs import FusionInputs
from explore_a.models.final_flow import ABLATIONS
from explore_a.training.losses import fusion_loss
from explore_a.training.prediction import make_fusion, predict
from explore_a.training.two_frame import reverse_pair, supervised, temporal_direction
from test_explore_a_geometry import example


@pytest.fixture
def pair():
    torch.set_num_threads(1)
    data = example(32, 48)
    data.update(gt=torch.full_like(data["current_depth"], 3.),
                previous_gt=torch.full_like(data["current_depth"], 3.2),
                gt_valid=torch.ones_like(data["current_depth"], dtype=torch.bool),
                previous_gt_valid=torch.ones_like(data["current_depth"], dtype=torch.bool),
                previous_image_valid=data["image_valid"].clone())
    return data


def test_self_reference_exact_zero_even_after_learning_and_soft_visibility(pair):
    config = load_config()
    model = make_fusion("final_flow", config)
    with torch.no_grad():
        model.increment_global.weight.normal_()
        model.increment_local.weight.normal_()
    pair["previous_depth"] = pair["current_depth"]
    geometry = pair_geometry(pair, config["geometry"])
    visible = torch.ones_like(pair["current_depth"]) * .5
    visible[..., :4, :4] = 0
    inputs = FusionInputs.from_pair(pair, geometry, visible)
    output = model(inputs)
    # grid_sample at zero displacement can introduce sub-ULP RGB/photo differences.
    torch.testing.assert_close(output["temporal_global_increment"],
                               torch.zeros_like(output["temporal_global_increment"]), atol=1e-5,
                               rtol=0)
    torch.testing.assert_close(output["temporal_local_increment"],
                               torch.zeros_like(output["temporal_local_increment"]), atol=1e-5,
                               rtol=0)


def test_s_is_history_independent_and_ablations_fix_blend_and_fallback(pair):
    config = load_config()
    model = make_fusion("final_flow", config)
    with torch.no_grad():
        for layer in (model.spatial_local, model.context.global_head,
                      model.increment_local, model.increment_global):
            layer.weight.normal_(std=.01)
    geometry = pair_geometry(pair, config["geometry"])
    inputs = FusionInputs.from_pair(pair, geometry, torch.ones_like(pair["gt"]))
    output = model(inputs)
    altered = replace(inputs, previous_rgb=inputs.previous_rgb * 0,
                       previous_depth=inputs.previous_depth * 7,
                       historical=inputs.historical * 4)
    other = model(altered)
    for key in ("spatial_global_logit", "spatial_local_logit"):
        torch.testing.assert_close(output[key], other[key], rtol=0, atol=0)
    for mode in ABLATIONS:
        changed = model.ablate(output, inputs, mode)
        assert changed["mixed_depth"] is output["mixed_depth"]
        assert changed["historical_weight"] is output["historical_weight"]
        assert changed["correction"].abs().max() <= .10000001
        assert changed["depth"].min() > 0
    absent = replace(inputs, availability=inputs.availability * 0)
    out = model(absent)
    for mode in ABLATIONS:
        torch.testing.assert_close(model.ablate(out, absent, mode)["depth"],
                                   pair["current_depth"], rtol=0, atol=0)
    assert not hasattr(model, "memory")


def test_reverse_is_involution_and_geometry_uses_previous_support(pair):
    pair["previous_image_valid"][..., :8] = False
    reverse = reverse_pair(pair)
    assert reverse["image_valid"] is pair["previous_image_valid"]
    twice = reverse_pair(reverse)
    for name in ("current_rgb", "previous_depth", "gt", "previous_gt", "forward", "backward"):
        torch.testing.assert_close(twice[name], pair[name], rtol=0, atol=0)
    geometry = pair_geometry(pair, load_config()["geometry"])
    assert not geometry["hard"][..., :8].any()


def test_temporal_target_allows_real_motion_and_gradients_reach_both_outputs(pair):
    pair["gt"], pair["previous_gt"] = pair["current_depth"], pair["previous_depth"]
    current = {"depth": pair["gt"].clone().requires_grad_()}
    previous = {"depth": pair["previous_gt"].clone().requires_grad_()}
    geometry = pair_geometry(pair, load_config()["geometry"])
    assert temporal_direction(current, previous, pair, geometry) == 0
    error = temporal_direction({"depth": current["depth"] * 1.1}, previous, pair, geometry)
    error.backward()
    assert current["depth"].grad.abs().sum() > 0
    assert previous["depth"].grad.abs().sum() > 0


def test_temporal_target_uses_warp_and_detaches_gt(pair):
    torch.manual_seed(2026)
    pair["forward"][:, 0] = 1
    pair["backward"][:, 0] = -1
    pair["previous_gt"] = (torch.rand_like(pair["previous_gt"]) + 2).requires_grad_()
    aligned, _ = warp(pair["previous_gt"].detach(), pair["backward"])
    pair["gt"] = (aligned * 1.05).requires_grad_()
    output = {"depth": pair["gt"].detach().clone().requires_grad_()}
    previous = {"depth": pair["previous_gt"].detach().clone().requires_grad_()}
    geometry = pair_geometry(pair, load_config()["geometry"])
    loss = temporal_direction(output, previous, pair, geometry)
    assert loss < 1e-12
    (loss + temporal_direction({"depth": output["depth"] * 1.01}, previous, pair,
                               geometry)).backward()
    assert pair["gt"].grad is None and pair["previous_gt"].grad is None


def test_bidirectional_missing_gt_and_temporal_warmup(pair):
    config = load_config()
    ramp = torch.linspace(0, 1, pair['current_depth'].shape[-1])[None,None,None,:]
    pair['current_depth'] = pair['current_depth'] + ramp
    pair['previous_depth'] = pair['previous_depth'] + ramp + .05*torch.sin(9*ramp)
    model = make_fusion("final_flow", config)
    output, terms = supervised(model, pair, config, "rule", 0)
    _, full = supervised(model, pair, config, "rule", 500)
    torch.testing.assert_close(full.total - terms.total, .05 * full.temporal)
    full.total.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.increment_local.weight.grad.abs().sum() > 0
    missing = {**pair, "previous_gt_valid": pair["previous_gt_valid"] & False,
               "previous_gt": pair["previous_gt"] * float("nan")}
    direct = fusion_loss(predict(model, missing, config, "rule")[0], missing, config["training"])
    _, averaged = supervised(model, missing, config, "rule", 500)
    torch.testing.assert_close(direct.total, averaged.total)
    assert averaged.temporal == 0
    empty = {**missing, "gt_valid": missing["gt_valid"] & False,
             "gt": missing["gt"] * float("nan")}
    _, terms = supervised(model, empty, config, "rule", 500)
    assert terms.total == 0
    assert FUSIONS == ("unet_guided", "flow_refine", "final_flow")


def test_shared_explicit_checkpoints_and_five_final_modes(tmp_path, fake_backbones):
    from test_explore_a_pipeline import online_experiment
    from test_explore_a_online import create_heads
    from explore_a.evaluation.final import compare
    config = online_experiment(tmp_path)
    create_heads(config)
    root = Path(config["paths"]["runs"])
    paths = [root / f"farneback-rule-{model}/best.pt"
             for model in ("unet_guided", "flow_refine", "final_flow")]
    before = [path.stat().st_mtime_ns for path in paths]
    depth, _, _ = fake_backbones
    results = compare(config, *paths, output=tmp_path / "comparison", device="cpu")
    assert len(results) == 8 and depth.calls == 3
    assert {r["ablation"] for r in results if r.get("fusion_model", {}).get(
        "architecture") == "final_flow"} == set(ABLATIONS)
    assert before == [path.stat().st_mtime_ns for path in paths]


def test_engine_final_training_resume_and_no_new_prediction_files(tmp_path, fake_backbones):
    from test_explore_a_pipeline import online_experiment
    from explore_a.training.engine import run_training
    from depth_project.io import read_json
    config = online_experiment(tmp_path)
    config["training"].update(steps=1, temporal_warmup_steps=1)
    config["training"]["sampling_mode"] = "video_sqrt"
    before = set(tmp_path.rglob("*.npz"))
    run_training(config, "farneback", fusion="final_flow", device="cpu")
    config["training"]["steps"] = 2
    run_training(config, "farneback", fusion="final_flow", device="cpu")
    path = Path(config["paths"]["runs"]) / "farneback-rule-final_flow"
    assert read_json(path / "complete.json")["steps"] == 2
    assert before == set(tmp_path.rglob("*.npz"))


def test_delivered_cli_arguments_and_final_config(monkeypatch, tmp_path):
    from explore_a.__main__ import main
    captured = []
    config_path = Path(__file__).parents[1] / "configs/explore_a/mixed_h5_da2_final.yaml"
    config = load_config(config_path)
    assert Path(config["paths"]["runs"]).name == "runs_da2_final_30k"
    assert config["training"]["steps"] == 30000
    monkeypatch.setattr("explore_a.training.engine.run_training", lambda *a: captured.append(a))
    monkeypatch.setattr("sys.argv", ["explore_a", "--config", str(config_path), "train",
                                    "--flow", "farneback", "--occlusion", "rule",
                                    "--fusion", "final_flow"])
    main()
    assert captured[-1][1:] == ("farneback", "rule", "final_flow")
    monkeypatch.setattr("explore_a.evaluation.final.compare", lambda *a: captured.append(a))
    args = ["explore_a", "--config", str(config_path), "evaluate-final"]
    for model in ("guided", "refine", "final"):
        args.extend(["--" + model + "-checkpoint", str(tmp_path / (model + ".pt"))])
    monkeypatch.setattr("sys.argv", [*args, "--output", str(tmp_path / "comparison")])
    main()
    assert captured[-1][1:4] == tuple(tmp_path / (m + ".pt")
                                    for m in ("guided", "refine", "final"))
    monkeypatch.setattr("explore_a.evaluation.part2.evaluate_part2",
                        lambda *a, **kw: captured.append(kw))
    monkeypatch.setattr("sys.argv", ["explore_a", "--config", str(config_path), "evaluate-part2",
                                    "--checkpoint", str(tmp_path / "final.pt"), "--ablations",
                                    "--setting", "short", "--output", str(tmp_path / "part2"),
                                    "--benchmark-root", str(tmp_path / "ResourcesLow/Datasets")])
    main()
    assert captured[-1]["ablations"] is True
    monkeypatch.setattr("explore_a.evaluation.engine.evaluate",
                        lambda *a, **kw: captured.append(kw))
    for mode in ABLATIONS:
        monkeypatch.setattr("sys.argv", ["explore_a", "--config", str(config_path), "evaluate",
                                        "--flow", "farneback", "--fusion", "final_flow",
                                        "--checkpoint", str(tmp_path / "final.pt"),
                                        "--output", str(tmp_path / mode), "--ablation", mode])
        main()
        assert captured[-1]["ablation"] == mode


from test_explore_a_pipeline import fake_backbones  # noqa: E402
