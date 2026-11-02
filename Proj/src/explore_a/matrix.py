"""Serial orchestration; each subprocess exits before the next GPU stage starts."""
import csv
from itertools import product
from pathlib import Path
import subprocess
import sys

from depth_project.io import read_json, write_json
from explore_a.config import FLOWS, FUSIONS, OCCLUSIONS, experiment_name, load_config


def command(config, action, *arguments):
    subprocess.run([sys.executable, "-m", "explore_a", "--config", config["config_path"],
                    action, *arguments], check=True)


def experiments():
    """Three DA heads, each with its own output root; both flow backends are explicit."""
    directory = Path(__file__).resolve().parents[2] / "configs/explore_a"
    return [(load_config(directory / filename), fusion) for filename, fusion in (
        ("mixed_h5_da2.yaml", "unet_guided"),
        ("mixed_h5_da2_refine.yaml", "flow_refine"),
        ("mixed_h5_da2_final.yaml", "final_flow"))]


def run_matrix(config, include_evaluation=False):
    for selected, fusion in experiments():
        for flow in FLOWS:
            command(selected, "train", "--flow", flow, "--occlusion", "rule", "--fusion", fusion)
            if include_evaluation:
                command(selected, "evaluate", "--flow", flow, "--fusion", fusion)


def summarize(config):
    root = Path(config["paths"]["runs"])
    records = []
    allowed = {experiment_name(*v) for v in product(FLOWS, OCCLUSIONS, FUSIONS)}
    for path in sorted(root.glob("*/evaluation-test/metrics.json")):
        result = read_json(path)
        if result["experiment"] not in allowed and not result["experiment"].startswith("baseline-"):
            continue
        for dataset, metrics in result["by_dataset"].items():
            records.append({"experiment": result["experiment"], "dataset": dataset, **metrics,
                            "online_warm_ms": result["latency"]["warm_ms"],
                            "fusion_warm_ms": result["latency"]["fusion_warm_ms"],
                            "latency_mode": result["latency"]["mode"],
                            **{f"input_{key}": value for key, value in
                               result["latency"].get("input_stats", {}).items()},
                            "peak_allocated_gib": result["latency"]["peak_allocated_gib"]})
    if not records:
        raise FileNotFoundError("No completed test evaluations to summarize")
    write_json(root / "summary.json", records)
    fields = sorted({key for record in records for key in record})
    with (root / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(root / "summary.csv")
