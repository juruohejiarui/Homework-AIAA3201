"""Infer image directories or prepared RGB videos with one selected model."""
import argparse
from pathlib import Path

import numpy as np

from depth_project.inference.engine import gpu_session, predict
from depth_project.io import read_rgb, write_json
from depth_project.models import load_model
from depth_project.paths import CHECKPOINTS

WEIGHTS = {
    "da2": "depth_anything_v2_vits.pth",
    "vda": "video_depth_anything_vits.pth",
    "depthpro": "depth_pro.pt",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=WEIGHTS, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--images", type=Path)
    inputs.add_argument("--frames", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--backend", choices=["upstream", "sdpa", "flash"], default="upstream")
    parser.add_argument("--metric", action="store_true", help="Use a VDA metric checkpoint")
    parser.add_argument("--focal", type=float, help="Depth Pro input focal length in pixels")
    args = parser.parse_args()
    if args.model == "vda" and args.images:
        parser.error("VDA requires consecutive video frames supplied with --frames")
    if args.frames:
        with np.load(args.frames) as source:
            frames = source["frames"]
            fps = float(source["fps"])
        names = [f"{index:06d}" for index in range(len(frames))]
    else:
        paths = sorted(
            path for path in args.images.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        frames = np.stack([read_rgb(path) for path in paths])
        names = [path.stem for path in paths]
        fps = 1
    weights = args.weights or CHECKPOINTS / WEIGHTS[args.model]
    with gpu_session():
        adapter = load_model(
            args.model, weights, args.precision == "fp32", args.input_size,
            args.backend, args.metric, args.focal,
        )
        depths, profile = predict(adapter, frames, fps)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.frames:
        np.savez_compressed(args.output / "depths.npz", depths=depths, fps=fps, kind=adapter.kind)
    else:
        for name, depth in zip(names, depths):
            np.save(args.output / f"{name}.npy", depth)
    profile.update(
        model=args.model, kind=adapter.kind, precision=args.precision,
        weights=str(weights), input_size=args.input_size,
    )
    if hasattr(adapter, "focal_predictions"):
        profile["focal_predictions_px"] = adapter.focal_predictions
    write_json(args.output / "profile.json", profile)
    print(args.output)


if __name__ == "__main__":
    main()
