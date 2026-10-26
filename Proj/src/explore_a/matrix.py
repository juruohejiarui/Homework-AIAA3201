"""Serial orchestration; each subprocess exits before the next GPU stage starts."""
import csv
from itertools import product
from pathlib import Path
import subprocess
import sys

from depth_project.io import read_json, write_json
from explore_a.config import FLOWS, FUSIONS, OCCLUSIONS


def command(config, action, *arguments):
    subprocess.run([sys.executable, "-m", "explore_a", "--config", config["config_path"],
                    action, *arguments], check=True)


def run_matrix(config, include_evaluation=True):
    for flow in FLOWS:
        command(config, "train-occlusion", "--flow", flow)
    for flow, occlusion, fusion in product(FLOWS, OCCLUSIONS, FUSIONS):
        arguments = ["--flow", flow, "--occlusion", occlusion, "--fusion", fusion]
        command(config, "train", *arguments)
    if include_evaluation:
        command(config, "evaluate-matrix")
        summarize(config)


def summarize(config):
    root = Path(config["paths"]["runs"])
    records = []
    for path in sorted(root.glob("*/evaluation-test/metrics.json")):
        result = read_json(path)
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
