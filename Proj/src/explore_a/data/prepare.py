"""Build an immutable raw-data manifest before online training or evaluation."""
from collections import Counter
from pathlib import Path

from depth_project.io import read_json, write_json
from depth_project.paths import PROJECT
from explore_a.config import bind_identity
from explore_a.data.history import evaluation_history
from explore_a.data.kitti import extract_training, prepare_kitti
from explore_a.data.nyu import prepare_nyu
from explore_a.data.scannet import prepare_scannet
from explore_a.data.splits import validate_rows
from explore_a.data.tum import prepare_tum

ADAPTERS = {"kitti": ("KITTI", prepare_kitti), "nyuv2": ("nyu_depth_v2", prepare_nyu),
            "scannet": ("ScanNet", prepare_scannet), "tum": ("TUM-RGBD", prepare_tum)}


def prepare(config, extract_kitti=False):
    path = Path(config["paths"]["manifest"])
    history = evaluation_history()
    snapshot = read_json(PROJECT / "configs/explore_a/historical_kitti.json")
    extra = set(history) - set(snapshot)
    if extra:
        print("Additional historical evaluation exclusions:", sorted(extra), flush=True)
    if path.exists():
        old = read_json(path)
        added = set(history) - set(old["history"])
        if added:
            print("New evaluation exclusions:", sorted(added), flush=True)
        raise ValueError("Manifest exists; preserve it and select NEW manifest/run paths")
    settings = config["data"]
    aliases = read_json(settings["scene_groups"]) if settings["scene_groups"] else {}
    root = Path(config["paths"]["datasets"])
    rows, reports = [], {}
    for dataset in settings["datasets"]:
        folder, adapter = ADAPTERS[dataset]
        if dataset == "kitti":
            if extract_kitti:
                extract_training(root / folder)
            selected, report = adapter(root / folder, settings, history, aliases.get(dataset, {}))
        else:
            selected, report = adapter(root / folder, settings, aliases.get(dataset, {}))
        if not selected:
            raise ValueError(f"No eligible pairs in requested dataset: {dataset}")
        for split in ("train", "dev"):
            if not any(row["split"] == split for row in selected):
                raise ValueError(f"Requested source has no eligible {split} pairs: {dataset}")
        rows.extend(selected)
        reports[dataset] = report
    validate_rows(rows, history)
    result = {"schema": 1, "settings": settings, "history": history, "scene_groups": aliases,
              "reports": reports, "counts": dict(Counter(
                  f"{r['dataset']}/{r['split']}" for r in rows)), "samples": rows,
              "test_label": "existing evaluation set; not an unseen benchmark"}
    bind_identity(path, result)
    write_json(path.with_name("historical_kitti_exclusions.json"), history)
    print("Manifest:", path, "\nCounts:", result["counts"], "\nReports:", reports, flush=True)


def load_manifest(config):
    from explore_a.data.history import assert_history_current

    result = read_json(config["paths"]["manifest"])
    if result["settings"] != config["data"]:
        raise ValueError("Data settings differ from the fixed manifest")
    assert_history_current(result)
    validate_rows(result["samples"], result["history"])
    return result
