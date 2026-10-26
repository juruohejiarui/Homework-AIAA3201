"""Frozen pairwise optical flow backends; neither path downloads weights."""
import cv2
import numpy as np
import torch

from explore_a.observability.progress import message


class PairFlow:
    def __init__(self, name, config, device="cuda"):
        self.name, self.device = name, device
        self.updates = config["geometry"]["raft_updates"]
        self.bidirectional_batch = config.get("online", {}).get(
            "raft_bidirectional_batch", True)
        self.model = None
        if name == "raft":
            from torchvision.models.optical_flow import raft_large

            self.model = raft_large(weights=None, progress=False).to(device).eval()
            self.model.load_state_dict(torch.load(config["weights"]["raft"], map_location="cpu",
                                                  weights_only=True))
            self.model.requires_grad_(False)

    @torch.inference_mode()
    def one_way(self, source, target):
        """RGB HWC uint8 -> flow[2,H,W] in source-grid pixels."""
        if self.name == "farneback":
            value = cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(source, cv2.COLOR_RGB2GRAY),
                cv2.cvtColor(target, cv2.COLOR_RGB2GRAY), None, .5, 3, 15, 3, 5, 1.2, 0)
            return value.transpose(2, 0, 1)
        if self.name != "raft":
            raise ValueError(self.name)
        tensors = [torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
                   .unsqueeze(0).to(self.device).float() / 127.5 - 1 for image in (source, target)]
        # RAFT's correlation lookup is kept FP32; frozen inference retains no backward graph.
        output = self.model(*tensors, num_flow_updates=self.updates)[-1]
        return output[0].float().cpu().numpy()

    @torch.inference_mode()
    def bidirectional(self, previous, current):
        """Batch order: previous->current, current->previous; units stay source-grid pixels."""
        images = torch.from_numpy(np.stack((previous, current)).transpose(0, 3, 1, 2).copy())
        images = images.to(self.device).float() / 127.5 - 1
        flow = self.model(images, images.flip(0), num_flow_updates=self.updates)[-1]
        result = flow.float().cpu().numpy()
        return result[0], result[1]

    def __call__(self, previous, current):
        retry = False
        if self.name == "raft" and self.bidirectional_batch:
            try:
                return self.bidirectional(previous, current)
            except torch.cuda.OutOfMemoryError:
                retry = True
        if retry:
            # Leave the exception block before freeing tensors retained by the traceback.
            torch.cuda.empty_cache()
            self.bidirectional_batch = False
            message("RAFT batch-2 OOM: use sequential directions for remaining pairs")
        return self.one_way(previous, current), self.one_way(current, previous)
