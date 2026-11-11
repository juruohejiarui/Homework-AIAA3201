"""Memory-only DA-V2/VDA Small reference evaluation on the frozen Part 2 video cohort."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from depth_project.data.protocol import select_sequences
from depth_project.inference.__main__ import WEIGHTS
from depth_project.inference.engine import predict
from depth_project.io import write_json
from depth_project.models import load_model
from depth_project.paths import CHECKPOINTS
from explore_a.evaluation.part2 import MANIFESTS, input_rgb, benchmark_root_for
from explore_a.evaluation.part2_protocol import (
    benchmark_path, official_score, metric_absrel, metric_scores,
)


@torch.inference_mode()
def full_image_predictions(adapter, sequence, dataset, setting, crop):
    """Keep predictions for full-sequence SSI, but never stack the entire RGB recording."""
    arrays = []
    for index, frame in enumerate(sequence["frames"]):
        rgb = input_rgb(frame, dataset, setting, crop)
        value = np.asarray(adapter.predict(rgb), np.float32)
        if value.shape != rgb.shape[:2] or not np.isfinite(value).all():
            raise ValueError("Invalid full-sequence reference prediction")
        arrays.append(value)
        if (index + 1) % 100 == 0:
            print("REFERENCE", dataset, sequence["name"], index + 1,
                  "/", len(sequence["frames"]), flush=True)
    return arrays, {"frames": len(arrays), "prediction_storage": "one full sequence in memory",
                    "timing": "not measured; full RGB sequence not retained"}


def run(model, output, benchmark_root, datasets=("bonn",), setting="long", tum_root=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {"model": model, "size": "Small", "precision": "fp32", "setting": setting,
                "benchmark_root": str(benchmark_root), "datasets": list(datasets),
                "resolution": "upstream; same actual DA-V2/VDA input-size rule",
                "weights": str(CHECKPOINTS / WEIGHTS[model]), "prediction_storage": "memory",
                "scoring": "v2; AC Sintel final image scores + video SSI + ordinal"}
    if model == "unidepthv2":
        from depth_project.models.unidepth import REVISION
        identity.update(revision=REVISION, resolution_level=9, intrinsics="RGB-only estimated",
                        resolution="official infer; resolution_level=9",
                        alignment="clip-level inverse SSI; separate metric AbsRel")
    receipt = output / "run.json"
    if receipt.exists() and json.loads(receipt.read_text()) != identity:
        raise ValueError("Reference output belongs to another evaluation")
    write_json(receipt, identity)
    adapter = None
    results = {}
    for dataset in datasets:
        dataset_root = benchmark_root_for(dataset, benchmark_root, tum_root)
        manifest = json.loads((dataset_root / MANIFESTS[dataset]).read_text())
        from explore_a.evaluation.sintel_ac import prepare_manifest, frame_mean
        prepare_manifest(manifest, dataset_root)
        sequences = (manifest["sequences"] if dataset == "sintel" else
                     select_sequences(manifest, "full" if dataset == "tum-validation" else setting))
        for sequence in sequences:
            for frame in sequence["frames"]:
                for key in ("image", "gt"):
                    frame[key] = benchmark_path(frame[key], dataset_root)
        write_json(output / dataset / "manifest.json", {**manifest, "sequences": sequences})
        records = []
        for sequence in sequences:
            path = output / dataset / sequence["name"] / "metrics.json"
            if path.exists():
                records.append(json.loads(path.read_text()))
                continue
            if adapter is None:
                adapter = load_model(model, CHECKPOINTS / WEIGHTS[model], fp32=True)
            if setting == "full":
                if hasattr(adapter, "predict_video"):
                    raise ValueError("Full TUM currently supports the requested image references")
                predictions, profile = full_image_predictions(
                    adapter, sequence, dataset, setting, manifest["config"]["crop"])
            else:
                images = np.stack([input_rgb(frame, dataset, setting, manifest["config"]["crop"])
                                   for frame in sequence["frames"]])
                ratio = max(images.shape[1:3]) / min(images.shape[1:3])
                adapter.input_size = (round(int(518 * 1.777 / ratio) / 14) * 14
                                      if model == "da2" and dataset != "sintel" and ratio > 1.78 else 518)
                if model == "da2":
                    adapter.align_corners = dataset != "sintel"
                predictions, profile = predict(adapter, images)
                del images
            if adapter.kind == "metric_m":
                if any((value <= 0).any() for value in predictions):
                    raise ValueError("Metric reference returned nonpositive depth")
                predictions = [1 / value for value in predictions]
            metrics = official_score(predictions, sequence["frames"], manifest["config"])
            if adapter.kind == "relative_inverse" or dataset == "sintel":
                from explore_a.evaluation.part2_protocol import aligned_sequence_scores
                metrics.update(aligned_sequence_scores(
                    predictions, sequence["frames"], manifest["config"],
                    f"{dataset}:{sequence['name']}"))
            if adapter.kind == "metric_m":
                if dataset == "tum-validation":
                    unaligned = metric_scores(predictions, sequence["frames"], manifest["config"])
                    metrics["metric_absrel_secondary"] = unaligned.get("abs_rel")
                    metrics.update({"metric_" + k: v for k, v in unaligned.items()})
                else:
                    metrics["metric_absrel_secondary"] = metric_absrel(
                        predictions, sequence["frames"], manifest["config"])
            record = {"sequence": sequence["name"], "frames": len(sequence["frames"]), **metrics,
                      "profile": profile}
            write_json(path, record)
            records.append(record)
            print("REFERENCE", model, dataset, sequence["name"], metrics, flush=True)
            del predictions
        result = {"dataset": dataset, "model": model, "setting": setting,
                  "sequences": records, "total_frames": sum(r["frames"] for r in records),
                  "mean": {key: float(np.mean([r[key] for r in records]))
                           for key in records[0] if key in ("abs_rel", "rmse", "delta1")
                           or key.endswith(("_img", "_video"))
                           or key in ("ordinal_accuracy", "ordinal_pairs")}}
        if dataset == "sintel":
            result["mean"].update(frame_mean(records))
            result["scoring_protocol"] = manifest["config"]["image_scoring"]
        if model == "unidepthv2":
            result["mean"]["metric_absrel_secondary"] = float(np.mean(
                [r["metric_absrel_secondary"] for r in records]))
            if dataset == "tum-validation":
                for key in records[0]:
                    if key.startswith("metric_") and key != "metric_absrel_secondary":
                        result["mean"][key] = float(np.mean([r[key] for r in records]))
        write_json(output / dataset / "metrics.json", result)
        results[dataset] = result
    write_json(output / "summary.json", results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("da2", "vda", "unidepthv2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--datasets", choices=tuple(MANIFESTS), nargs="+", default=["bonn"])
    parser.add_argument("--setting", choices=("short", "long", "full"), default="long")
    args = parser.parse_args()
    run(args.model, args.output, args.benchmark_root, args.datasets, args.setting)


if __name__ == "__main__":
    main()
