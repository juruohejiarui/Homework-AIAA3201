"""Save a few review panels, with one shared depth range per panel."""
import cv2
import numpy as np


def save_panel(path, data, output, maximum):
    rgb = data["current_rgb"][0].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    panels = [rgb[:, :, ::-1]]
    image_valid = data["image_valid"][0, 0].cpu().numpy().astype(bool)
    for index, value in enumerate((data["gt"], data["current_depth"], output["depth"])):
        depth = value[0, 0].detach().cpu().numpy()
        encoded = np.clip(depth / maximum * 255, 0, 255).astype(np.uint8)
        panel = cv2.applyColorMap(encoded, cv2.COLORMAP_TURBO)
        panel[~image_valid] = 0
        if index == 0:
            panel[~data["gt_valid"][0, 0].cpu().numpy().astype(bool)] = 0
        panels.append(panel)
    for key in ("historical_weight", "visibility"):
        value = output[key][0, 0].detach().float().cpu().numpy()
        panels.append(cv2.applyColorMap(np.clip(value * 255, 0, 255).astype(np.uint8),
                                        cv2.COLORMAP_VIRIDIS))
    labels = ("RGB", "GT", "Depth Pro", "Output", "History weight", "Availability")
    for panel, label in zip(panels, labels):
        cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.concatenate(panels, 1)):
        raise OSError(path)
