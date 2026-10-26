"""Numerical invariants that protect scientific meaning, without model downloads or CUDA."""
import numpy as np
import pytest
import torch

from explore_a.data.reading import fit_depth, resize_intrinsics
from explore_a.geometry.features import fusion_features, occlusion_features
from explore_a.geometry.warp import pair_geometry, warp
from explore_a.models.correlation import CompiledSampler, TorchCorrelation
from explore_a.models.fusion import Fusion


def example(height=16, width=24):
    rgb = torch.randint(0, 255, (1, 3, height, width), dtype=torch.uint8)
    return {"previous_rgb": rgb, "current_rgb": rgb.clone(),
            "previous_depth": torch.full((1, 1, height, width), 4.),
            "current_depth": torch.full((1, 1, height, width), 2.),
            "image_valid": torch.ones(1, 1, height, width, dtype=torch.bool),
            "forward": torch.zeros(1, 2, height, width),
            "backward": torch.zeros(1, 2, height, width)}

SETTINGS = {"fb_relative": .01, "fb_absolute_px2": .5}


def test_sampling_direction_and_out_of_bounds():
    ramp = torch.arange(8).float().reshape(1, 1, 1, 8).expand(1, 1, 4, 8)
    backward = torch.zeros(1, 2, 4, 8)
    backward[:, 0] = -2
    sampled, valid = warp(ramp, backward)
    torch.testing.assert_close(sampled[:, :, :, 2:], ramp[:, :, :, :-2])
    assert not valid[:, :, :, :2].any()


def test_feature_channels_and_fb_mask():
    data = example()
    geometry = pair_geometry(data, SETTINGS)
    assert geometry["rule"].all()
    assert fusion_features(data, geometry, geometry["hard"].float()).shape[1] == 12
    assert occlusion_features(data, geometry).shape[1] == 13
    data["forward"][:, 0] = 5
    assert not pair_geometry(data, SETTINGS)["rule"].any()


def test_soft_visibility_renormalizes_both_confidences():
    data = example()
    model = Fusion("cnn")
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    geometry = pair_geometry(data, SETTINGS)
    output = model(data, geometry, torch.ones_like(data["current_depth"]) * .25)
    # Equal confidence (.5,.5) and history availability .25 give .125/(.5+.125)=.2.
    torch.testing.assert_close(output["historical_weight"], torch.full_like(data["current_depth"], .2))
    torch.testing.assert_close(output["depth"], torch.full_like(data["current_depth"], 2.4))


@pytest.mark.parametrize("architecture", ["cnn", "unet", "residual"])
def test_absent_history_is_exact_fallback_and_gradient_is_finite(architecture):
    data = example()
    geometry = pair_geometry(data, SETTINGS)
    model = Fusion(architecture)
    absent = model(data, geometry, torch.zeros_like(data["current_depth"]))
    torch.testing.assert_close(absent["depth"], data["current_depth"])
    assert not absent["historical_weight"].any()
    present = model(data, geometry, torch.ones_like(data["current_depth"]))
    present["depth"].mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if architecture == "residual":
        assert present["residual_fraction"].abs().max() <= .1
    else:
        assert present["depth"].min() >= 2 and present["depth"].max() <= 4


def test_sparse_gt_splat_and_intrinsic_pixel_centers():
    depth = np.zeros((8, 8), np.float32)
    depth[1, 1], depth[2, 2] = 4, 3
    result = fit_depth(depth, (4, 8))
    assert (result > 0).sum() <= 2
    assert result.max() <= 4 and (result == 0).any()
    matrix = np.array([[100., 0, 4], [0, 100., 4], [0, 0, 1]])
    resized = resize_intrinsics(matrix, (8, 8), (4, 8))
    np.testing.assert_allclose(resized, [[50, 0, 3.75], [0, 50, 1.75], [0, 0, 1]])


def test_correlation_sum_and_offset_order():
    left = torch.ones(1, 3, 4, 5)
    right = torch.arange(20).reshape(1, 1, 4, 5).float().expand_as(left)
    actual = TorchCorrelation(patch_size=3)(left, right)
    torch.testing.assert_close(actual[:, 1, 1], (left * right).sum(1))
    torch.testing.assert_close(actual[:, 1, 2, :, :-1], 3 * right[:, 0, :, 1:])
    assert not actual[:, 1, 2, :, -1].any()


@pytest.mark.skipif(CompiledSampler is None, reason="optional compiled correlation not installed")
def test_optional_compiled_correlation_matches_fallback():
    left, right = torch.randn(1, 8, 8, 8), torch.randn(1, 8, 8, 8)
    compiled = CompiledSampler(kernel_size=1, patch_size=7, stride=1, padding=0, dilation=1)
    torch.testing.assert_close(compiled(left, right), TorchCorrelation()(left, right))
