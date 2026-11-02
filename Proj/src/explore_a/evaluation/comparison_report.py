"""Collect finished, protocol-separated four-model comparisons without running inference."""
import argparse
import json
from pathlib import Path

from depth_project.io import write_json


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def report(output):
    output = Path(output)
    rows = []

    def add(scope, dataset, model, metrics, alignment):
        if metrics:
            metrics = {key.removeprefix("ssi_"): value for key, value in metrics.items()}
            rows.append({"scope": scope, "dataset": dataset, "model": model,
                         "alignment": alignment, **metrics})

    for scope in ("dev", "dev-full", "test"):
        root = output / scope
        for path in sorted((root / "fusion").glob("*/evaluation-*/metrics.json")):
            result = read(path)
            for source, metrics in result["by_dataset"].items():
                relative = result.get("frontend", {}).get("name") == "da2"
                alignment = "per-image inverse SSI" if relative else "none"
                add(scope, source, result["experiment"], metrics, alignment)
        for model in ("unidepthv2", "da2"):
            result = read(root / model / "metrics.json")
            if result:
                for source, metrics in result["by_dataset"].items():
                    add(scope, source, model, metrics, result["alignment"])
    scopes = [(output / "benchmark", "legacy-video", False),
              (output / "benchmark/vda-short", "VDA-short-appendix", False),
              (output / "benchmark/vda-long", "VDA-long-selection", False),
              (output / "tum-validation", "legacy-TUM-truncated", True),
              (output / "tum-validation/full-official-validation", "TUM-full-official", True)]
    for root, label, tum in scopes:
        for folder in ("fusion", "unidepthv2", "da2", "vda"):
            result = read(root / folder / "summary.json")
            if not result:
                continue
            for source, values in result.items():
                models = values["mean"] if folder == "fusion" else {folder: values["mean"]}
                for model, metrics in models.items():
                    scope = f"{label}-{values.get('setting', 'unknown')}"
                    add(scope + "-SSI", source, model, metrics, "clip inverse SSI")
                    if tum:
                        metric = {k.removeprefix("metric_"): v for k, v in metrics.items()
                                  if k.startswith("metric_") and k != "metric_absrel_secondary"}
                        add(scope + "-metric", source, model, metric, "none")
    root = output / "nyu"
    result = read(root / "fusion/metrics.json")
    if result:
        for model, metrics in result["mean"].items():
            alignment = result.get("alignment", "none")
            scope = "official-labeled-SSI" if alignment != "none" else "official-labeled"
            add(scope, "nyuv2-654", model, metrics, alignment)
    for model in ("unidepthv2", "da2"):
        result = read(root / model / "metrics.json")
        if result:
            add("official-labeled", "nyuv2-654", model, result["mean"], result["alignment"])
    write_json(output / "comparison.json", rows)
    lines = ["# Fixed-checkpoint model comparison", "",
             "Fusion frontends DA-V2-S FP16 via cache/DGX/local; head follows saved AMP; "
             "independent references retain their evaluation precision.",
             "DA2 GT-assisted results are separate from RGB-only metric results.",
             "NYU labeled has no history: head equals the same-canvas DA-V2-S control.",
             "Video SSI is an additional research protocol, not the course metric ranking.", "",
             "VDA long/short rows have different input clips and must not be merged.",
             "Sintel uses float32 meter GT; Bonn uses timestamp association. These documented",
             "decoding fixes differ from flawed public extraction code; exact paper-cohort",
             "identity is not established. TUM full is not a recovered UniDepth paper list.", "",
             "| Scope | Dataset | Model | Alignment | AbsRel ↓ | RMSE ↓ | δ1 ↑ |",
             "|---|---|---|---|---:|---:|---:|"]
    for row in rows:
        delta = row.get("d1", row.get("delta1", float("nan")))
        lines.append(f"| {row['scope']} | {row['dataset']} | {row['model']} | "
                     f"{row['alignment']} | {row['abs_rel']:.6f} | {row['rmse']:.6f} | "
                     f"{delta:.6f} |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    print(output / "comparison.md", flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    report(parser.parse_args().output)
