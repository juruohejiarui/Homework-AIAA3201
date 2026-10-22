"""Evaluate complete video predictions with upstream inverse-depth SSI metrics."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from depth_project.data.protocol import SETTINGS, select_sequences
from depth_project.evaluation.official import load_evaluator
from depth_project.io import read_json, write_json


def evaluate_sequences(manifest, setting, predictions, tae=False, device="cuda"):
    config = manifest["config"]
    top, bottom, left, right = config["crop"]
    arguments = SimpleNamespace(a=top, b=bottom, c=left, d=right, max_eval_len=500,
                                max_depth_eval=config["max_depth"], hard_crop=True, mask=False)
    evaluator = load_evaluator(tae, device)
    rows = []
    for sequence in select_sequences(manifest, setting):
        frames = sequence["frames"]
        paths = [predictions / sequence["name"] / f"{index:06d}.npy" for index in range(len(frames))]
        missing = next((path for path in paths if not path.is_file()), None)
        if missing:
            raise FileNotFoundError(f"Incomplete sequence prediction: {missing}")
        ground_truth = [frame["gt"] for frame in frames]
        factors = [config["factor"]] * len(frames)
        paths = [str(path) for path in paths]
        if tae:
            cameras = [np.array(frame["K"]) for frame in frames]
            poses = [np.array(frame["pose"]) for frame in frames]
            score = evaluator.eval_TAE(paths, ground_truth, factors, [], cameras, poses, arguments)
            metrics = {"tae_percent": float(score)}
        else:
            values = evaluator.eval_depthcrafter(paths, ground_truth, factors, arguments)
            metrics = dict(zip(["abs_rel", "rmse", "delta1"], map(float, values)))
        rows.append({"sequence": sequence["name"], "frames": len(frames), **metrics})
    means = {key: float(np.mean([row[key] for row in rows]))
             for key in rows[0] if key not in {"sequence", "frames"}}
    return {"dataset": manifest["dataset"], "setting": setting, "sequences": rows,
            "mean": means, "total_frames": sum(row["frames"] for row in rows),
            "protocol": manifest.get("protocol"),
            "tae_note": "Original K/hard-crop and collision convention; output times100" if tae else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--setting", choices=[name for name in SETTINGS if name != "tae"], default="long")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    result = evaluate_sequences(read_json(args.manifest), args.setting, args.predictions, device=args.device)
    write_json(args.output, result)
    print(result["mean"])


if __name__ == "__main__":
    main()
