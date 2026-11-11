"""Complete fixed continuous-dev oracle with resumable per-clip inference staging."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from depth_project.io import read_json, write_json
from explore_a.config import file_identity, load_config
from explore_a.data.prepare import load_manifest
from explore_a.data.reading import depth_range
from explore_a.evaluation.depth import MetricSummary
from explore_a.evaluation.oracle_scoring import BRANCHES, score_clip
from explore_a.frontend import metadata
from explore_a.geometry.relative import affine
from explore_a.online.factory import make_runtime
from explore_a.online.loading import OnlinePairs
from explore_a.training.dev_subset import select_dev, selection_record
from explore_a.training.prediction import fusion_geometry


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



def stage_path(output, row):
    from hashlib import sha256
    return output / "staging" / (sha256(row["id"].encode()).hexdigest()+".npy")


def summarize(rows, records, protocol, clips):
    summaries, diagnostics = {}, MetricSummary()
    for row in rows:
        record = records[row["id"]]
        for category in ("primary", "primary_video", "secondary", "secondary_video"):
            for name, metrics in record[category].items():
                summary = summaries.setdefault(category+"/"+name, MetricSummary())
                count = record["common_pixels"] if category.startswith("primary") else (
                    record["valid_pixels"])
                summary.add(row, metrics, count)
        diagnostics.add(row, record["diagnostics"], record["valid_pixels"])
    completed = {}
    for key, summary in summaries.items():
        has_frames = any(v["valid_frames"] for v in summary.counts.values())
        completed[key] = (summary.finish() if has_frames
                          else {"unavailable": "empty mask", "by_dataset": {}, "macro_mean": {}})
    gains = {}
    for scope in ("primary", "primary_video"):
        raw = completed[scope+"/raw"]
        for name in BRANCHES:
            branch = completed[scope+"/"+name]
            gains[scope+"/"+name] = {"by_dataset": {
                source: values["inverse_loss"]-branch["by_dataset"][source]["inverse_loss"]
                for source, values in raw["by_dataset"].items() if "inverse_loss" in values}}
    for clip in clips:
        selected = [records[sample] for sample in clip["sample_ids"]]
        clip["branches"] = {}
        for category in ("primary", "primary_video", "secondary", "secondary_video"):
            for name in BRANCHES:
                values = [record[category][name] for record in selected
                          if record[category][name] is not None]
                keys = {key for value in values for key in value}
                clip["branches"][category+"/"+name] = {
                    key: float(np.mean([v[key] for v in values if key in v])) for key in keys}
    descriptions = {
        "primary": "Image baseline-fixed gauge; per-pixel oracle targets in this gauge",
        "primary_video": "Clip baseline-fixed gauge; per-pixel oracle targets in this gauge",
        "secondary": "Image-gauge oracle predictions, independently refitted img/video scores",
        "secondary_video": "Clip-gauge oracle predictions, independently refitted img/video scores"}
    return {"protocol": protocol, "branches": completed, "primary_gains": gains,
            "category_descriptions": descriptions,
            "diagnostics": diagnostics.finish(), "clips": clips,
            "notice": "Conditional output-range bounds, not network learnability; secondary "
                      "refitted image/video scores are not strict bounds"}


def markdown(result):
    lines = ["# DA continuous dev oracle", "",
             f"Flow: {result['protocol']['flow']}; occlusion: rule.", "", result["notice"], "",
             "| Fixed gauge / branch | KITTI | NYUv2 | TUM | Macro inverse loss |",
             "|---|---:|---:|---:|---:|"]
    for key, value in result["branches"].items():
        if key.startswith("primary") and not value.get("unavailable"):
            columns = [value["by_dataset"].get(s, {}).get("inverse_loss", float("nan"))
                       for s in ("kitti", "nyuv2", "tum")]
            lines.append(f"| {key} | " + " | ".join(f"{v:.6f}" for v in columns)
                         + f" | {value['macro_mean'].get('inverse_loss', float('nan')):.6f} |")
    lines += ["", "Full fixed-map depth metrics, secondary dual alignment, ordinal scores,",
              "per-sequence/per-clip metrics and coverage are in summary.json and clips/*.json."]
    return "\n".join(lines)+"\n"


@torch.no_grad()
def run(config, output, device="cuda", flow="farneback"):
    if flow not in ("farneback", "raft"):
        raise ValueError(f"Unsupported oracle flow: {flow}")
    rows = select_dev([r for r in load_manifest(config)["samples"] if r["split"] == "dev"],
                      config["training"]["dev_sample_fraction"], config["data"]["seed"])
    output = Path(output)
    cohort = selection_record(rows)
    protocol = {"schema": 2, "frontend": metadata(config), **cohort,
                "counts": dict(Counter(r["dataset"] for r in rows)),
                "manifest": file_identity(config["paths"]["manifest"]),
                "weights": file_identity(config["weights"]["da2"]),
                "data": config["data"], "geometry": config["geometry"],
                "flow": flow, "occlusion": "rule", "fraction": .1,
                "score": "baseline-fixed image and clip gauges; common mask + full GT; "
                         "secondary independently refitted image/video SSI + raw ordinal"}
    if flow == "raft":
        protocol["flow_weights"] = file_identity(config["weights"]["raft"])
    path = output / "protocol.json"
    if path.exists() and read_json(path) != protocol:
        raise ValueError("Oracle output belongs to another input/score protocol")
    write_json(path, protocol)
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("evaluation_clip", row["sequence"])].append(row)
    records, receipts, pending = {}, [], []
    for i, (clip, selected) in enumerate(groups.items()):
        receipt = output / "clips" / f"{i:04d}.json"
        if receipt.exists():
            saved = read_json(receipt)
            if saved["sample_ids"] != [r["id"] for r in selected]:
                raise ValueError("Oracle clip IDs differ")
            records.update({r["id"]: r for r in saved["records"]})
            receipts.append({k: v for k, v in saved.items() if k != "records"})
            for row in selected:
                stage_path(output, row).unlink(missing_ok=True)
                stage_path(output, row).with_suffix(".json").unlink(missing_ok=True)
        else:
            pending.append((i, clip, selected))
    remaining = [r for _, _, selected in pending for r in selected
                 if not (stage_path(output, r).exists() and
                         stage_path(output, r).with_suffix(".json").exists())]
    runtime, iterator = None, None
    from explore_a.observability.progress import Progress
    try:
        if remaining:
            runtime = make_runtime(config, [flow], False, device)
            dataset = OnlinePairs(config, remaining, runtime, flow, phase="dev")
            iterator = dataset.iter_batches()
        with Progress(config, "DA DEV ORACLE", len(rows), len(rows)-len(remaining)) as progress:
            if iterator is not None:
                for indices, data in iterator:
                    geometry = fusion_geometry(data, config)
                    for local, index in enumerate(indices):
                        row = remaining[index]
                        packed = torch.cat((data["current_depth"][local],
                                            geometry["historical"][local],
                                            geometry["unaligned_historical"][local],
                                            data["gt"][local], data["gt_valid"][local].float(),
                                            geometry["rule"][local].float(),
                                            torch.ones_like(data["gt"][local])*
                                            geometry["field_scale"][local]), dim=0)
                        path = stage_path(output, row)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = path.with_suffix(".part")
                        with temporary.open("wb") as stream:
                            np.save(stream, packed.float().cpu().numpy(), allow_pickle=False)
                        temporary.replace(path)
                        write_json(path.with_suffix(".json"), {
                            "alignment_accepted": float(geometry["alignment_accepted"][local]),
                            "alignment_scale": float(geometry["alignment_scale"][local]),
                            "alignment_shift": float(geometry["alignment_shift"][local])})
                        progress.update()
        # Frozen inference is complete before CPU scoring; staging permits interruption recovery.
        if iterator is not None:
            iterator.close()
            iterator = None
        if runtime is not None:
            runtime.close()
            runtime = None
        for i, clip, selected in pending:
            paths = [stage_path(output, r) for r in selected]
            measured, mapping = score_clip(selected, paths,
                depth_range(selected[0]["dataset"], config["data"]))
            for record, stage in zip(measured, paths):
                record["diagnostics"].update(read_json(stage.with_suffix(".json")))
            saved = {"clip": clip, "sample_ids": [r["id"] for r in selected],
                     "video_gauge": mapping, "records": measured}
            # Commit a complete clip before removing its resumable CPU staging.
            write_json(output / "clips" / f"{i:04d}.json", saved)
            receipts.append({k: v for k, v in saved.items() if k != "records"})
            records.update({r["id"]: r for r in measured})
            for stage in paths:
                stage.unlink()
                stage.with_suffix(".json").unlink(missing_ok=True)
            print("ORACLE CLIP", i+1, "/", len(groups), selected[0]["dataset"], flush=True)
    finally:
        if iterator is not None:
            iterator.close()
        if runtime is not None:
            runtime.close()
    result = summarize(rows, records, protocol, sorted(receipts, key=lambda r: r["clip"]))
    write_json(output / "summary.json", result)
    (output / "summary.md").write_text(markdown(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--flow", choices=("farneback", "raft"), default="farneback")
    args = parser.parse_args()
    run(load_config(args.config), args.output, args.device, flow=args.flow)


if __name__ == "__main__":
    main()
