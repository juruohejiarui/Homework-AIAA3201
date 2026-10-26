"""Portable, full-precision long-form scalar table derived from the valid log history."""
import csv
import math
from pathlib import Path
import tempfile

from depth_project.io import read_json
from explore_a.observability.history import development
from explore_a.observability.scalars import baseline_scalars, scalars

FIELDS = ("step", "phase", "scope", "dataset", "sequence", "camera", "metric", "value", "tag")


def tagged(step, tag, value):
    parts = tag.split("/")
    scope = parts[1] if len(parts) == 3 else "macro"
    dataset = "" if scope in ("macro", "best") else scope
    return {"step": step, "phase": parts[0], "scope": "dataset" if dataset else scope,
            "dataset": dataset, "sequence": "", "camera": "", "metric": parts[-1],
            "value": value, "tag": tag}


def rows(history, baseline):
    for kind in ("train", "dev", "test"):
        records = development(history) if kind == "dev" else history[kind]
        for record in records:
            for tag, value in scalars(kind, record):
                if math.isfinite(float(value)):
                    yield tagged(record["step"], tag, value)
            if kind == "test":
                for sequence in record.get("sequences", []):
                    for phase, values in (("test", sequence["metrics"]),
                                          ("test_counts", sequence["counts"])):
                        for metric, value in values.items():
                            yield {"step": record["step"], "phase": phase, "scope": "sequence",
                                   "dataset": sequence["dataset"], "sequence": sequence["sequence"],
                                   "camera": sequence["camera"], "metric": metric, "value": value,
                                   "tag": f"{phase}/sequence/{sequence['dataset']}/{metric}"}
    for tag, value in baseline_scalars(baseline):
        yield tagged(0, tag, value)


def export_csv(root, history):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "heldout_baseline.json"
    baseline = read_json(path) if path.exists() else None
    with tempfile.NamedTemporaryFile(mode="w", newline="", dir=root, suffix=".csv",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows(history, baseline))
    try:
        temporary.replace(root / "scalars.csv")
    finally:
        temporary.unlink(missing_ok=True)
