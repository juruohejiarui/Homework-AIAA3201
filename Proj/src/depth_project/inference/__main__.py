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
    "unidepthv2": "unidepth-v2-vits14",
}


def predict_images(adapter, paths, output, batch_size):
    seconds = 0.0
    peak_gpu_bytes = 0
    for start in range(0, len(paths), batch_size):
        batch = paths[start:start + batch_size]
        depths, profile = predict(adapter, np.stack([read_rgb(path) for path in batch]))
        for path, depth in zip(batch, depths, strict=True):
            np.save(output / f"{path.stem}.npy", depth)
        seconds += profile["seconds"]
        peak_gpu_bytes = max(peak_gpu_bytes, profile["peak_gpu_bytes"])
        if start == 0 or (start // batch_size + 1) % 10 == 0 or start + len(batch) == len(paths):
            print(f"Saved {start + len(batch)}/{len(paths)} depth maps", flush=True)
    profile.update(frames=len(paths), seconds=seconds, seconds_per_frame=seconds / len(paths),
                   peak_gpu_bytes=peak_gpu_bytes, image_batch_size=batch_size)
    return profile


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
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--backend", choices=["upstream", "sdpa", "flash"], default="upstream")
    parser.add_argument("--metric", action="store_true", help="Use a VDA metric checkpoint")
    args = parser.parse_args()
    if args.model == "vda" and args.images:
        parser.error("VDA requires consecutive video frames supplied with --frames")
    if args.image_batch_size < 1:
        parser.error("--image-batch-size must be positive")
    if args.frames:
        with np.load(args.frames) as source:
            frames = source["frames"]
            fps = float(source["fps"])
    else:
        paths = sorted(
            path for path in args.images.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not paths:
            parser.error(f"No input images in {args.images}")
    weights = args.weights or CHECKPOINTS / WEIGHTS[args.model]
    args.output.mkdir(parents=True, exist_ok=True)
    with gpu_session():
        adapter = load_model(
            args.model, weights, args.precision == "fp32", args.input_size,
            args.backend, args.metric,
        )
        if args.frames:
            depths, profile = predict(adapter, frames, fps)
            np.savez_compressed(args.output / "depths.npz", depths=depths, fps=fps, kind=adapter.kind)
        else:
            profile = predict_images(adapter, paths, args.output, args.image_batch_size)
    profile.update(
        model=args.model, kind=adapter.kind, precision=args.precision,
        weights=str(weights), input_size=args.input_size,
    )
    write_json(args.output / "profile.json", profile)
    print(args.output)


if __name__ == "__main__":
    main()
