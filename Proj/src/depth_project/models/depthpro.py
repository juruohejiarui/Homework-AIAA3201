"""Depth Pro adapter, including optional known focal length in input pixels."""
from dataclasses import replace

import torch


class DepthProAdapter:
    kind = "metric_m"

    def __init__(self, weights, fp32=True, focal=None):
        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

        config = replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(weights))
        precision = torch.float32 if fp32 else torch.float16
        self.model, self.transform = depth_pro.create_model_and_transforms(
            config=config, device=torch.device("cuda"), precision=precision,
        )
        self.model.eval()
        self.focal = None if focal is None else torch.tensor(focal, device="cuda")
        self.focal_predictions = []

    def predict(self, rgb):
        prediction = self.model.infer(self.transform(rgb), f_px=self.focal)
        self.focal_predictions.append(float(prediction["focallength_px"].detach().cpu()))
        return prediction["depth"].detach().float().cpu().numpy()
