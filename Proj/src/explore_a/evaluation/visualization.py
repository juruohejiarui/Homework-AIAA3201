"""Save a few review panels, with one shared depth range per panel."""
import cv2
import numpy as np


def save_panel(path, data, output, maximum):
    rgb = data["current_rgb"][0].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    # RGB -> BGR slicing has a negative channel stride; OpenCV drawing needs
    # writable, contiguous pixels (even when imwrite would accept the view).
    panels = [np.ascontiguousarray(rgb[:, :, ::-1])]
    image_valid = data["image_valid"][0, 0].cpu().numpy().astype(bool)
    relative = output.get("prediction_kind") == "relative_inverse"
    if relative:
        maximum = max(float(np.quantile(data["current_depth"][0, 0].cpu().numpy()[image_valid],
                                        .98)), 1e-6)
    for index, value in enumerate((data["gt"], data["current_depth"], output["depth"])):
        depth = value[0, 0].detach().cpu().numpy()
        if relative and index == 0:
            depth = 1/np.maximum(depth, 1e-6)
            depth = depth/max(float(np.quantile(depth[data["gt_valid"][0, 0].cpu().numpy()],
                                                .98)), 1e-6)*maximum if (
                data["gt_valid"][0, 0].any()) else np.zeros_like(depth)
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
    labels = ("RGB", "GT", "DA-V2-S", "Output", "History weight", "Availability")
    if relative:
        labels = ("RGB", "GT inverse (own range)", "DA2 inverse", "Output inverse",
                  "History weight", "Availability")
    for panel, label in zip(panels, labels):
        cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.concatenate(panels, 1)):
        raise OSError(path)
