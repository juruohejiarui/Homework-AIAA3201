"""Exercise the actual UOM inference graph using synthetic backbones, without weights/network."""
import torch
from torchvision.models import resnet50

from explore_a.models.mftiq import UOM_ARGUMENTS
from explore_a.vendor.mftiq.UOM.features import pretrained_features_extractor_baseline4 as features
from explore_a.vendor.mftiq.UOM.network.uom_baseline4 import UOMNetBase4


class SyntheticDino(torch.nn.Module):
    patch_size = 14

    def get_intermediate_layers(self, image):
        tokens = image.shape[-2] // 14 * (image.shape[-1] // 14)
        return [image.new_zeros((image.shape[0], tokens, 384))]


def test_uom_offline_inference_shapes_and_frozen_graph(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(features, "load_resnet", lambda paths: resnet50(weights=None))
    monkeypatch.setattr(features, "load_dinov2", lambda paths: SyntheticDino())
    model = UOMNetBase4(**UOM_ARGUMENTS, offline_paths={}).eval().requires_grad_(False)
    image = torch.ones(1, 3, 64, 80) * 128
    flow = torch.zeros(1, 2, 64, 80)
    with torch.inference_mode():
        outputs = model(image, image, flow, inference_mode=True,
                        binary_uncertainty_construction=True)
    assert outputs["occlusion"].shape == (1, 64, 80)
    assert torch.isfinite(outputs["uncertainty"]).all()
    assert outputs["occlusion"].min() >= 0 and outputs["occlusion"].max() <= 1
    assert not any(parameter.requires_grad for parameter in model.parameters())
