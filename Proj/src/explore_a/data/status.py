"""Report configured sources separately from samples actually prepared and runs completed."""
from pathlib import Path

from depth_project.io import read_json


def dataset_status(config):
    manifest = Path(config["paths"]["manifest"])
    result = {"config": config["config_path"], "configured_datasets": config["data"]["datasets"],
              "nyuv2_training_source": ("FastDepth H5, adjacent retained frames; unknown time gap"
                                        if config["data"].get("nyu_h5_root") else
                                        "full eligible Raw sequences, not labeled-only"),
              "sampling": f"all eligible pairs in pool; {config['training']['steps']}-step "
                          "replacement sampling does not guarantee visiting every frame",
              "source_weights": config["training"].get("source_weights", "equal per dataset"),
              "execution": config.get("execution", {}).get("backend", "online"),
              "manifest": str(manifest), "manifest_exists": manifest.is_file(),
              "prepared_datasets": [], "prepared_counts": {}, "completed_training_runs": 0}
    if manifest.exists():
        prepared = read_json(manifest)
        result["prepared_datasets"] = sorted({row["dataset"] for row in prepared["samples"]})
        result["prepared_counts"] = prepared["counts"]
        result["manifest_settings_match"] = prepared["settings"] == config["data"]
    runs = Path(config["paths"]["runs"])
    result["completed_training_runs"] = len(list(runs.glob("*/complete.json")))
    return result
