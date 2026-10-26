"""Read-only crash audit: inventories first, optional NPZ decoding for affected jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

from depth_project.io import read_json, write_json
from explore_a.cache.paths import component
from explore_a.cache.plan import load_job
from explore_a.cache.settings import LOCAL_ROOT, RECIPE, root_path
from explore_a.cache.storage import read_npz, safe_relative


def inspect_job(root, name, deep=False):
    root = Path(root)
    job = load_job(root, name)
    receipt = root / "jobs" / name / "local.complete.json"
    result = {"job": name, "complete": receipt.exists(), "deep": deep, "files": 0,
              "bytes": 0, "missing": [], "bad": [], "metadata_errors": []}
    sizes = {}
    if receipt.exists():
        try:
            report = read_json(receipt)
            sizes = report["sizes"]
            if not isinstance(sizes, dict):
                raise ValueError("Completion sizes must be a file inventory")
            if (report["recipe"] != job["recipe"] or report["job"] != name or
                    set(sizes) != set(job["outputs"]) or sum(sizes.values()) != report["bytes"]):
                raise ValueError("Completion receipt disagrees with fixed job inventory")
        except (ValueError, KeyError, TypeError) as error:
            result["metadata_errors"].append(str(error))
            sizes = {}
    spec = read_json(root / "recipe.json")
    dataset = job["rows"][0]["dataset"]
    canvas = spec["data"]["canvas_kitti" if dataset == "kitti" else "canvas_indoor"]
    for relative in job["outputs"]:
        path = root / safe_relative(relative)
        try:
            if path.is_symlink():
                raise ValueError("Artifact is a symlink")
            if path.exists() and not path.is_file():
                raise ValueError("Artifact is not a regular file")
            size = path.stat().st_size
            if relative in sizes and size != sizes[relative]:
                raise ValueError(f"Size differs: {size} != {sizes[relative]}")
            if deep:
                values = read_npz(path)  # Reads ZIP CRC, dtype and finite array values.
                channels = {"forward": 2, "backward": 2} if relative.startswith("flow/") else {
                    "depth" if relative.startswith("frames/") else "visibility": 1}
                if set(values) != set(channels):
                    raise ValueError(f"Unexpected NPZ fields: {list(values)}")
                for key, count in channels.items():
                    if values[key].shape != (count, *canvas):
                        raise ValueError(f"Unexpected {key} shape: {values[key].shape}")
            result["files"] += 1
            result["bytes"] += size
        except FileNotFoundError:
            result["missing"].append(relative)
        except (OSError, ValueError, EOFError, KeyError) as error:
            result["bad"].append({"path": relative, "error": str(error)})
        except Exception as error:
            # zipfile/zlib exceptions also indicate unreadable output, not a failed GPU model.
            result["bad"].append({"path": relative, "error": repr(error)})
    return result


def inspect(root, names=None, deep=False, workers=4):
    root = Path(root)
    if names is None:
        names = sorted(path.parent.name for path in (root / "jobs").glob("*/local.complete.json"))
    names = [component(name) for name in names]
    results, started = [], time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(lambda name: inspect_job(root, name, deep), names):
            results.append(result)
            print(f"AUDIT {len(results)}/{len(names)} {result['job']}: "
                  f"files={result['files']} missing={len(result['missing'])} "
                  f"bad={len(result['bad'])} metadata={len(result['metadata_errors'])}", flush=True)
    return {"root": str(root), "deep": deep, "seconds": time.monotonic() - started,
            "jobs": results, "files": sum(item["files"] for item in results),
            "bytes": sum(item["bytes"] for item in results),
            "bad_files": sum(len(item["bad"]) for item in results),
            "missing_files": sum(len(item["missing"]) for item in results),
            "metadata_errors": sum(len(item["metadata_errors"]) for item in results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root_path(LOCAL_ROOT, RECIPE))
    parser.add_argument("--job", action="append")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    result = inspect(args.root, args.job, args.deep, args.workers)
    write_json(args.output, result)
    print({key: value for key, value in result.items() if key != "jobs"}, flush=True)


if __name__ == "__main__":
    main()
