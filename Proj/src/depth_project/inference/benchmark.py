"""Infer a frozen benchmark manifest; evaluation is a separate command."""
import argparse
from pathlib import Path

import numpy as np

from depth_project.data.protocol import SETTINGS, select_sequences
from depth_project.inference.engine import gpu_session, predict
from depth_project.inference.__main__ import WEIGHTS
from depth_project.io import read_json, read_rgb, write_json
from depth_project.models import load_model
from depth_project.paths import CHECKPOINTS


def sequence_images(sequence, dataset, setting, crop):
    top, bottom, left, right = crop
    images = []
    for frame in sequence["frames"]:
        image = read_rgb(frame["image"])
        if setting != "tae" and dataset in {"scannet", "nyuv2"}:
            image = image[top:bottom, left:right]
        elif setting != "tae" and dataset in {"kitti", "bonn"}:
            height, width = image.shape[:2]
            image = image[:height // 2 * 2, :width // 2 * 2]
        images.append(image)
    return np.stack(images)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", choices=["vda", "da2"], required=True)
    parser.add_argument("--setting", choices=SETTINGS, default="long")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    sequences = select_sequences(manifest, args.setting)
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {"manifest": manifest, "model": args.model,
                "setting": args.setting, "precision": args.precision}
    run_path = args.output / "run.json"
    if run_path.exists() and read_json(run_path) != identity:
        raise ValueError("Output directory belongs to a different experiment")
    write_json(run_path, identity)
    with gpu_session():
        adapter = load_model(args.model, CHECKPOINTS / WEIGHTS[args.model], args.precision == "fp32")
        for sequence in sequences:
            output = args.output / sequence["name"]
            complete = output / "complete.json"
            if complete.exists():
                print(f"Reuse {sequence['name']}", flush=True)
                continue
            images = sequence_images(sequence, manifest["dataset"], args.setting, manifest["config"]["crop"])
            if args.model == "da2":
                ratio = max(images.shape[1:3]) / min(images.shape[1:3])
                adapter.input_size = round(int(518 * 1.777 / ratio) / 14) * 14 if ratio > 1.78 else 518
            depths, profile = predict(adapter, images)
            output.mkdir(exist_ok=True)
            for index, depth in enumerate(depths):
                np.save(output / f"{index:06d}.npy", depth)
            profile["source_indices"] = [frame["source_index"] for frame in sequence["frames"]]
            write_json(complete, profile)
            print(sequence["name"], profile["frames"], profile["seconds"], flush=True)


if __name__ == "__main__":
    main()
