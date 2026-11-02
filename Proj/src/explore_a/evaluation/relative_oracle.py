"""DA conditional bounds on the complete fixed dev subset, not a learned method result."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import torch

from depth_project.io import read_json, write_json
from depth_project.paths import OUTPUTS
from explore_a.config import file_identity, load_config
from explore_a.data.prepare import load_manifest
from explore_a.evaluation.depth import MetricSummary, field_scores
from explore_a.frontend import metadata, relative
from explore_a.geometry.relative import affine
from explore_a.online.factory import make_runtime
from explore_a.online.loading import OnlinePairs
from explore_a.training.dev_subset import select_dev
from explore_a.training.prediction import fusion_geometry
from explore_a.training.relative import gauge, main_loss


@torch.no_grad()
def candidates(data, geometry, fixed, fraction=.1):
    current, history = data["current_depth"].float(), geometry["historical"]
    raw_history, truth = geometry["unaligned_historical"], fixed["target"]
    common = geometry["rule"]

    def project(lower, upper):
        value = torch.maximum(lower.clamp_min(0), torch.minimum(truth, upper.clamp_min(0)))
        return torch.where(common, value, current)

    low, high = torch.minimum(current, history), torch.maximum(current, history)
    raw_low, raw_high = torch.minimum(current, raw_history), torch.maximum(current, raw_history)
    scale = geometry["field_scale"]
    first, last = 2*current-history, 2*history-current
    a, b, valid = affine(raw_history, truth, common & fixed["valid"], minimum=2, robust=False)
    gt_history = (a*raw_history+b).clamp_min(0)
    diagnostic = project(torch.minimum(current, gt_history), torch.maximum(current, gt_history))
    diagnostic = torch.where(valid, diagnostic, current)
    return {
        "raw": current,
        "half_unaligned": torch.where(common, .5*(current+raw_history), current),
        "half_aligned": torch.where(common, .5*(current+history), current),
        "oracle_unaligned": project(raw_low, raw_high),
        "oracle_aligned": project(low, high),
        "oracle_additive10": project(low-fraction*scale, high+fraction*scale),
        "oracle_signed_m1_p2": project(torch.minimum(first, last), torch.maximum(first, last)),
        "gt_alignment_diagnostic": diagnostic,
    }


def run(config, output, device="cuda"):
    if not relative(config):
        raise ValueError("DA oracle requires frontend.name: da2")
    config = copy.deepcopy(config)
    full = [r for r in load_manifest(config)["samples"] if r["split"] == "dev"]
    rows = select_dev(full, config["training"]["dev_sample_fraction"], config["data"]["seed"])
    output = Path(output)
    protocol = {"schema": 1, "frontend": metadata(config), "sample_ids": [r["id"] for r in rows],
                "counts": dict(Counter(r["dataset"] for r in rows)),
                "manifest": file_identity(config["paths"]["manifest"]),
                "weights": file_identity(config["weights"]["da2"]),
                "data": config["data"], "geometry": config["geometry"],
                "backend": "unified frozen pipeline; one shared pass; no prediction writes",
                "score": "Fixed baseline->GT gauge, common GT/FB/aligned-history mask; "
                         "secondary full-GT per-image SSI with exact fallback",
                "fraction": .1}
    path = output / "protocol.json"
    if path.exists() and read_json(path) != protocol:
        raise ValueError("Oracle output belongs to another input/score protocol")
    write_json(path, protocol)
    log, records = output / "frames.jsonl", {}
    if log.exists():
        with log.open("rb+") as reader:
            while line := reader.readline():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if reader.read(1):
                        raise ValueError("Corrupt oracle record before final line")
                    reader.truncate(reader.tell()-len(line))
                    break
                records[record["id"]] = record
    if set(records) - {r["id"] for r in rows}:
        raise ValueError("Oracle log contains IDs outside the fixed dev cohort")
    remaining = [r for r in rows if r["id"] not in records]
    if remaining:
        runtime = make_runtime(config, ["farneback"], False, device)
        dataset = OnlinePairs(config, remaining, runtime, "farneback", phase="dev")
        iterator = dataset.iter_batches()
        from explore_a.observability.progress import Progress
        try:
            with log.open("a", buffering=1) as stream, torch.no_grad(), \
                    Progress(config, "DA DEV ORACLE", len(rows), len(records)) as progress:
                for indices, data in iterator:
                    geometry, fixed = fusion_geometry(data, config), gauge(data)
                    predictions = candidates(data, geometry, fixed)
                    for local, index in enumerate(indices):
                        row = remaining[index]
                        valid = fixed["valid"][local]
                        fixed_i = {k: v[local:local+1] for k, v in fixed.items()}
                        common = valid & geometry["rule"][local]
                        fixed_i["valid"] = common[None]
                        primary, secondary = {}, {}
                        for name, prediction in predictions.items():
                            loss = main_loss(prediction[local:local+1], fixed_i)
                            primary[name] = {"inverse_loss": float(loss)} if common.any() else None
                            secondary[name] = field_scores(prediction[local], data["gt"][local],
                                                           valid, config, row["dataset"])
                        if common.any():
                            values = {k: v["inverse_loss"] for k, v in primary.items()}
                            if (values["oracle_aligned"] > values["raw"]+1e-6 or
                                    values["oracle_additive10"] > values["oracle_aligned"]+1e-6):
                                raise AssertionError("Oracle reachable-set nesting violated")
                        use = valid & geometry["rule"][local]
                        truth = fixed["target"][local]
                        current = data["current_depth"][local]
                        history = geometry["historical"][local]
                        denom = use.sum().clamp_min(1)
                        above, below = (current > truth) & (history > truth), (
                            current < truth) & (history < truth)
                        projected = predictions["oracle_additive10"][local]
                        low, high = torch.minimum(current, history), torch.maximum(current, history)
                        span = .1*geometry["field_scale"][local]
                        touches = (projected <= (low-span).clamp_min(0)+1e-6) | (
                            projected >= high+span-1e-6)
                        diagnostics = {
                            "rule_coverage": float(use.sum()/valid.sum().clamp_min(1)),
                            "alignment_accepted": float(geometry["alignment_accepted"][local]),
                            "alignment_scale": float(geometry["alignment_scale"][local]),
                            "alignment_shift": float(geometry["alignment_shift"][local]),
                            "both_high": float((use & above).sum()/denom),
                            "both_low": float((use & below).sum()/denom),
                            "inside_interval": float((use & ~(above | below)).sum()/denom),
                            "correction_boundary": float((use & touches).sum()/denom)}
                        record = {"id": row["id"], "primary": primary, "secondary": secondary,
                                  "common_pixels": int(common.sum()),
                                  "valid_pixels": int(valid.sum()), "diagnostics": diagnostics}
                        stream.write(json.dumps(record, allow_nan=False)+"\n")
                        records[row["id"]] = record
                        progress.update()
        finally:
            iterator.close()
            runtime.close()
    summaries, diagnostics = {}, MetricSummary()
    for row in rows:
        record = records[row["id"]]
        for category in ("primary", "secondary"):
            for name, metrics in record[category].items():
                summary = summaries.setdefault(category+"/"+name, MetricSummary())
                count = record["common_pixels"] if category == "primary" else record["valid_pixels"]
                summary.add(row, metrics, count)
        diagnostics.add(row, record["diagnostics"], record["valid_pixels"])
    completed = {}
    for key, summary in summaries.items():
        if any(value["valid_frames"] for value in summary.counts.values()):
            completed[key] = summary.finish()
        else:
            completed[key] = {"by_dataset": {}, "macro_mean": {}, "unavailable": "empty mask"}
    result = {"protocol": protocol, "branches": completed,
              "diagnostics": diagnostics.finish(),
              "notice": "Conditional oracle only; secondary refitted SSI is not a strict bound"}
    raw = completed["primary/raw"]
    gains = {}
    if not raw.get("unavailable"):
        base = {tuple(r[k] for k in ("dataset", "sequence", "camera")): r["metrics"]
                for r in raw["sequences"]}
        for name, summary in completed.items():
            if not name.startswith("primary/") or summary.get("unavailable"):
                continue
            sequences = []
            for row in summary["sequences"]:
                key = tuple(row[k] for k in ("dataset", "sequence", "camera"))
                if "inverse_loss" in row["metrics"] and "inverse_loss" in base[key]:
                    sequences.append({"dataset": key[0], "sequence": key[1], "camera": key[2],
                        "loss_reduction": base[key]["inverse_loss"]-row["metrics"]["inverse_loss"]})
            gains[name] = {"sequences": sequences, "by_dataset": {
                source: raw["by_dataset"][source]["inverse_loss"]-metrics["inverse_loss"]
                for source, metrics in summary["by_dataset"].items() if "inverse_loss" in metrics}}
    result["primary_gains"] = gains
    write_json(output / "summary.json", result)
    lines = ["# DA dev oracle", "", "Full fixed subset; conditional bounds, not learned results.",
             "", "| Branch | KITTI | TUM | NYUv2 | Macro inverse loss |",
             "|---|---:|---:|---:|---:|"]
    for key, value in result["branches"].items():
        if key.startswith("primary/") and not value.get("unavailable"):
            columns = [value["by_dataset"].get(s, {}).get("inverse_loss", float("nan"))
                       for s in ("kitti", "tum", "nyuv2")]
            lines.append(f"| {key} | " + " | ".join(f"{v:.6f}" for v in columns)
                         + f" | {value['macro_mean']['inverse_loss']:.6f} |")
    (output / "summary.md").write_text("\n".join(lines)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    run(load_config(args.config), args.output, args.device)


if __name__ == "__main__":
    main()
