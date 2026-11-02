"""Prepare a separate sensor-depth video benchmark from official validation recordings."""
import argparse
from pathlib import Path

from depth_project.data.association import associate_tum
from depth_project.io import write_json
from depth_project.paths import DATASETS
from explore_a.download.tum_validation import validation_plan


def prepare(plan_path, output, tolerance=.02, max_gap=.1):
    plan = validation_plan(plan_path)
    root = Path(plan["output"]) / "sequences"
    sequences = []
    for entry in plan["files"]:
        name = entry["name"].removesuffix(".tgz")
        scene = root / name
        if not (scene / "rgb.txt").is_file() or not (scene / "depth.txt").is_file():
            raise FileNotFoundError(f"Download/extract the validation sequence first: {scene}")
        frames, last = [], None
        for image, depth, association in associate_tum(scene, tolerance):
            if not image.is_file() or not depth.is_file():
                raise FileNotFoundError(f"Missing associated RGB/depth: {image}; {depth}")
            now = association["rgb_timestamp"]
            frames.append({"image": str(image.resolve()), "gt": str(depth.resolve()),
                           **association, "reset_history": last is None or now - last > max_gap})
            last = now
        if len(frames) < 2:
            raise ValueError(f"Insufficient associated validation frames: {name}")
        sequences.append({"name": name, "frames": frames})
    result = {"version": 1, "dataset": "tum-validation", "sequences": sequences,
              "config": {"folder": "TUM-RGBD/official_validation", "factor": 5000,
                         "crop": [0, 480, 0, 640], "max_depth": 10},
              "protocol": {"source_plan": str(plan_path.resolve()),
                           "selection": "All official validation recordings in the fixed plan",
                           "association": "one-to-one nearest RGB/depth within tolerance",
                           "sync_tolerance_s": tolerance, "history_reset_gap_s": max_gap,
                           "depth": "registered sensor depth; uint16 / 5000 meters",
                           "notice": "Project depth evaluation, not TUM trajectory submission; "
                                     "not an exact reproduction of UniDepth's unpublished list"}}
    if output.exists():
        from depth_project.io import read_json
        if read_json(output) != result:
            raise ValueError("Existing TUM benchmark manifest differs; choose another output")
    else:
        write_json(output, result)
    print("TUM VALIDATION", len(sequences), "sequences;",
          sum(len(s["frames"]) for s in sequences), "associated frames;", output)
    return result


def main():
    from explore_a.download.tum_validation import DEFAULT_MANIFEST

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path,
                        default=Path.home() / "Documents/Resources/Datasets/TUM-RGBD"
                        / "official_validation"
                        / "prepared/manifest.json")
    args = parser.parse_args()
    prepare(args.plan, args.output)


if __name__ == "__main__":
    main()
