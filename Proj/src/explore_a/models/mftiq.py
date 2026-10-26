"""Frozen independent MFTIQ UOM; evaluate the selected flow without replacing it."""
import torch

UOM_ARGUMENTS = {
    "uom_features": 32, "uom_use_group_norm": True, "single_logits": True,
    "downsample_coef": 4, "head_type": "uncthreshold_stack",
    "with_spatial_correlation_sampler": True, "upsample_type": "bilinear",
    "with_resnet": True, "with_imgcnn": True, "with_dinov2": True, "with_vggnet": False,
}


class MFTIQVisibility:
    def __init__(self, config, device="cuda"):
        from explore_a.vendor.mftiq.UOM.network.uom_baseline4 import UOMNetBase4

        self.device = device
        paths = {**config["weights"], "dinov2_source": config["paths"]["dinov2_source"]}
        self.model = UOMNetBase4(**UOM_ARGUMENTS, offline_paths=paths)
        state = torch.load(config["weights"]["mftiq"], map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, previous, current, backward):
        """Return 1-occlusion on the CURRENT grid; RGB inputs are HWC uint8, flow is 2HW."""
        images = [torch.as_tensor(image.transpose(2, 0, 1).copy(), device=self.device)
                  .unsqueeze(0).float() for image in (current, previous)]
        flow = torch.as_tensor(backward.copy(), device=self.device).unsqueeze(0).float()
        outputs = self.model(*images, flow, inference_mode=True,
                             binary_uncertainty_construction=True)
        return {"visibility": (1 - outputs["occlusion"]).float().cpu().numpy(),
                "uncertainty": outputs["uncertainty"].float().cpu().numpy()}
