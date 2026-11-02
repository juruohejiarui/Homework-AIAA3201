"""Recover missing fixed RGB/GT manifests only when adapters exactly match cached job membership."""
import argparse
from collections import Counter
from pathlib import Path

from depth_project.io import read_json, write_json
from explore_a.config import load_config
from explore_a.data.history import assert_history_current
from explore_a.data.kitti import prepare_kitti
from explore_a.data.tum import prepare_tum
from explore_a.data.splits import validate_rows


def recover(config, cache_root):
    path, root = Path(config["paths"]["manifest"]), Path(cache_root)
    if path.exists():
        raise ValueError(f"Manifest already exists; refusing to overwrite: {path}")
    plan = read_json(root / "plan.json")
    history = plan["history"]
    assert_history_current({"history": history})
    fixed = {}
    sources = {}
    for item in plan["jobs"]:
        folder = root / "jobs" / item["job"]
        job = read_json(folder / "job.json")
        if job["rows"][0]["dataset"] not in config["data"]["datasets"]:
            continue
        mapping = read_json(folder / "sources.json")
        for row in job["rows"]:
            if row["id"] in fixed:
                raise ValueError(f"Duplicate cached sample: {row['id']}")
            fixed[row["id"]] = row
            sources[row["id"]] = {prefix: mapping[row[prefix + "_rgb"]]["source"]
                                   for prefix in ("previous", "current")}
    aliases = read_json(config["data"]["scene_groups"]) if config["data"]["scene_groups"] else {}
    dataset_root = Path(config["paths"]["datasets"])
    selected, reports = [], {}
    for name in config["data"]["datasets"]:
        if name == "kitti":
            rows, report = prepare_kitti(dataset_root / "KITTI", config["data"], history,
                                         aliases.get(name, {}))
        elif name == "tum":
            rows, report = prepare_tum(dataset_root / "TUM-RGBD", config["data"],
                                       aliases.get(name, {}))
        else:
            raise ValueError("Recovery only supports the original KITTI/TUM manifest")
        selected.extend(rows)
        reports[name] = report
    current = {r["id"]: r for r in selected}
    if current.keys() != fixed.keys() or len(current) != len(selected):
        raise ValueError(f"Fixed membership changed: {len(fixed.keys() - current.keys())} missing, "
                         f"{len(current.keys() - fixed.keys())} added; no manifest was written")
    for key, row in current.items():
        if any(row[k] != v for k, v in fixed[key].items() if not k.endswith("_rgb")):
            raise ValueError(f"Fixed split/pair/group changed: {key}")
        if any(Path(row[p + "_rgb"]).resolve() != Path(sources[key][p]).resolve()
               for p in ("previous", "current")):
            raise ValueError(f"Fixed RGB source changed: {key}")
    validate_rows(selected, history)
    result = {"schema": 1, "settings": config["data"], "history": history,
              "scene_groups": aliases, "samples": selected, "reports": reports,
              "counts": dict(Counter(f"{r['dataset']}/{r['split']}" for r in selected)),
              "test_label": "existing evaluation set; not an unseen benchmark"}
    write_json(path, result)
    print(f"Recovered {len(selected)} fixed pairs: {path}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    recover(load_config(args.config), args.cache_root.expanduser())


if __name__ == "__main__":
    main()
