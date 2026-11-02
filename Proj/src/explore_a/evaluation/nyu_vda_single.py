"""VDA Small image benchmark: each labeled image is a separate B=1,T=1 model input."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from depth_project.evaluation.metrics import align_inverse_depth
from depth_project.io import read_rgb, write_json
from depth_project.models import load_model
from depth_project.paths import CHECKPOINTS
from explore_a.evaluation.nyu_labeled import metric_function, score


@torch.inference_mode()
def run(root, output):
    root, output = Path(root), Path(output)
    ids = json.loads((root / "test/manifest.json").read_text())["test_ids"]
    adapter = load_model("vda", CHECKPOINTS / "video_depth_anything_vits.pth", fp32=False)
    from video_depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
    from torchvision.transforms import Compose
    import cv2
    transform = Compose([
        Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True,
               ensure_multiple_of=14, resize_method="lower_bound",
               image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[.485, .456, .406], std=[.229, .224, .225]), PrepareForNet()])
    records, metric = [], metric_function()
    for image_id in ids:
        name = f"{image_id:04d}"
        receipt = output / "images" / (name + ".json")
        if receipt.exists():
            records.append(json.loads(receipt.read_text()))
            continue
        rgb = read_rgb(root / "test/rgb" / (name + ".png"))
        value = transform({"image": rgb.astype(np.float32) / 255})["image"]
        tensor = torch.from_numpy(value).unsqueeze(0).unsqueeze(0).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            # Call the supported single-frame forward directly. The video wrapper would
            # pad to 32 frames, which is not the paper's single-frame input experiment.
            prediction = adapter.model(tensor)
        assert prediction.shape[:2] == (1, 1)
        prediction = functional.interpolate(prediction.float(), size=rgb.shape[:2],
                                            mode="bilinear", align_corners=True)[0, 0]
        inverse = prediction.cpu().numpy().astype(np.float64)
        truth = np.load(root / "test/gt" / (name + ".npy")).astype(np.float64)
        mask = np.zeros(truth.shape, bool)
        mask[45:471, 41:601] = True
        mask &= np.isfinite(truth) & (truth > .001) & (truth < 10)
        aligned, diagnostic = align_inverse_depth(inverse, truth, mask)
        record = {"image_id": image_id, "metrics": score(aligned, truth, metric),
                  "constrained_solution": diagnostic["constrained_to_constant"]}
        write_json(receipt, record)
        records.append(record)
        if len(records) % 50 == 0:
            print("VDA SINGLE IMAGE", len(records), "/", len(ids), flush=True)
    result = {"model": "VDA-S", "precision": "fp16", "input": "B=1,T=1; no image-to-image state",
              "input_size": 518, "count": len(records), "per_image": records,
              "alignment": "GT-assisted constrained inverse SSI, same as DA-V2 reference",
              "mean": {key: float(np.mean([r["metrics"][key] for r in records]))
                       for key in records[0]["metrics"]},
              "constrained_solutions": sum(r["constrained_solution"] for r in records),
              "note": "Small image-input comparison; paper Table 2 uses Large, not reproduced"}
    write_json(output / "metrics.json", result)
    print("VDA SINGLE COMPLETE", result["mean"], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)


if __name__ == "__main__":
    main()
