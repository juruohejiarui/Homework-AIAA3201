"""Evaluate a snapshotted fusion head on existing Part 2 video manifests."""
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from depth_project.data.protocol import select_sequences
from depth_project.io import read_rgb, write_json
from explore_a.data.reading import fit_image
from explore_a.evaluation.part2_protocol import (
    benchmark_path, metric_absrel, metric_scores, official_score, restore_canvas,
)
from explore_a.online.factory import make_runtime
from explore_a.online.video import video_raw
from explore_a.training.prediction import make_fusion, predict
from explore_a.models.final_flow import ABLATIONS, FinalFlow
from explore_a.models.inputs import FusionInputs

MANIFESTS = {"kitti": "KITTI/prepared/vda_protocol_v2/manifest.json",
             "sintel": "MPI-Sintel/prepared/vda_protocol_v1/manifest.json",
             "bonn": "Bonn-RGBD/prepared/vda_protocol_v1/manifest.json",
             "tum-validation": "TUM-RGBD/official_validation/prepared/manifest.json"}


def benchmark_root_for(dataset, benchmark_root, tum_root=None):
    if dataset != "tum-validation":
        return Path(benchmark_root)
    # TUM lives beside training data; other benchmarks remain on ResourcesLow.
    root = Path(tum_root).expanduser() if tum_root else (
        Path.home() / "Documents/Resources/Datasets/TUM-RGBD")
    if root.name != "TUM-RGBD":
        raise ValueError("--tum-root must point to the TUM-RGBD dataset directory")
    return root.parent


def raw_input(previous, current, dataset, sequence, before, now, config):
    canvas = config["data"]["canvas_kitti" if dataset == "kitti" else "canvas_indoor"]
    images = {True: previous, False: current}
    fitted, data = {}, {}
    for old, prefix in ((True, "previous"), (False, "current")):
        fitted[old], valid = fit_image(images[old], canvas)
        data[prefix + "_rgb"] = torch.from_numpy(
            fitted[old].transpose(2, 0, 1).copy())
        data[prefix + "_image_valid"] = torch.from_numpy(valid[None])
        if not old:
            data["image_valid"] = torch.from_numpy(valid[None])
    return {"row": {"dataset": dataset, "sequence": sequence, "camera": "part2",
                     "previous_id": before, "current_id": now,
                     "id": f"{dataset}:{sequence}:{now}"},
            "images": images, "fitted": fitted, "canvas": canvas, "data": data}


def input_rgb(frame, dataset, setting, crop):
    rgb = read_rgb(frame["image"])
    if dataset in {"scannet", "nyuv2"}:
        top, bottom, left, right = crop
        rgb = rgb[top:bottom, left:right]
    elif dataset in {"kitti", "bonn"}:
        height, width = rgb.shape[:2]
        rgb = rgb[:height // 2 * 2, :width // 2 * 2]
    return rgb


def evaluate_part2(config, checkpoint, output, benchmark_root, setting="long",
                   datasets=("kitti", "sintel", "bonn"), runtime=None, ablations=False,
                   tum_root=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "checkpoint.pt"
    if snapshot.exists():
        state = torch.load(snapshot, map_location="cpu", weights_only=False)
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = {key: state[key] for key in ("model", "identity", "step", "best")}
        torch.save(state, snapshot)
    print("Part 2 evaluation: pinned checkpoint step", state["step"],
          "setting", setting, "benchmark root", benchmark_root, flush=True)
    task = state["identity"]["experiment"]
    from explore_a.training.identity import check_frontend
    from explore_a.frontend import relative, metadata, valid_prediction
    check_frontend(state["identity"], config)
    inverse_frontend = relative(config)
    flow = task["flow"]
    config = copy.deepcopy(config)
    config["training"].update(state["identity"]["training"])
    model = make_fusion(task["fusion"], config).cuda().eval()
    from explore_a.training.checkpoint import load_model_state
    load_model_state(model, state["model"])
    if ablations and not isinstance(model, FinalFlow):
        raise ValueError("Part 2 correction ablations require a final_flow checkpoint")
    identity = {"checkpoint_step": state["step"], "task": task, "setting": setting,
                "datasets": list(datasets), "benchmark_root": str(benchmark_root),
                "precision": "DA-V2-S fp16; fusion follows checkpoint AMP setting",
                "execution": "blocking frozen components v2; cache/DGX/local",
                "protocol": "v2; AC Sintel final image scores; whole-video SSI retained",
                "input": "full selected images under Part2 crop; existing trained fusion canvas",
                "recurrence": "raw previous depth; no recurrent output state; first frame raw",
                "provenance": "independent benchmark root; no training manifest changes"}
    if inverse_frontend:
        identity["frontend"] = metadata(config)
        identity["precision"] = "DA2 fp16; fusion follows checkpoint AMP; relative inverse"
        identity["relative_protocol"] = (
            "Official whole-sequence SSI per branch; separate baseline-fixed whole-video gauge; "
            "no per-frame normalization")
    if ablations:
        identity["ablations"] = list(ABLATIONS)
    identity_path = output / "run.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Part 2 output belongs to a different evaluation")
    write_json(identity_path, identity)
    runtime = runtime or make_runtime(config, [flow], task["occlusion"] == "mftiq")
    results = {}
    try:
        for dataset in datasets:
            dataset_root = benchmark_root_for(dataset, benchmark_root, tum_root)
            path = dataset_root / MANIFESTS[dataset]
            manifest = json.loads(path.read_text())
            from explore_a.evaluation.sintel_ac import prepare_manifest, frame_mean
            prepare_manifest(manifest, dataset_root)
            selected_setting = "full" if dataset == "tum-validation" else setting
            sequences = manifest["sequences"] if dataset == "sintel" else select_sequences(manifest, selected_setting)
            # Keep an explicit relocated copy; original frozen manifests are never overwritten.
            for sequence in sequences:
                for frame in sequence["frames"]:
                    for key in ("image", "gt"):
                        frame[key] = benchmark_path(frame[key], dataset_root)
            write_json(output / dataset / "manifest.json", {**manifest, "sequences": sequences})
            records = []
            for sequence in sequences:
                receipt = output / dataset / sequence["name"] / "metrics.json"
                if receipt.exists():
                    records.append(json.loads(receipt.read_text()))
                    continue
                frames = sequence["frames"]
                names = ["raw_native", "raw_canvas", "fusion"]
                if ablations:
                    names.extend("fusion-" + mode for mode in ABLATIONS[1:])
                arrays = {name: [] for name in names}
                from explore_a.online.statistics import snapshot, difference
                input_before = snapshot(runtime)
                began = time.monotonic()
                decoded = video_raw(sequence, dataset, setting, manifest["config"]["crop"],
                                    config, runtime, flow=flow,
                                    teacher=task["occlusion"] == "mftiq")
                for i, raw, has_history in decoded:
                    current = raw["images"][False]
                    with torch.no_grad():
                        base = runtime.depth_pair(raw, ordered=True)
                        # Native baseline keeps details discarded by the fusion canvas.
                        native = raw["native_predictions"]["current"].copy()
                        canvas_raw = base["current_depth"][0].float().cpu().numpy()
                        fused = canvas_raw
                        alternatives = {}
                        if has_history:
                            data = runtime.add_flow(raw, base, flow)
                            data = {k: v.unsqueeze(0) for k, v in data.items()}
                            amp = bool(config["training"]["amp"])
                            with torch.autocast("cuda", enabled=amp):
                                predicted, geometry = predict(
                                    model, data, config, task["occlusion"])
                                if ablations:
                                    inputs = FusionInputs.from_pair(
                                        data, geometry, torch.ones_like(data["current_depth"]))
                                    for mode in ABLATIONS[1:]:
                                        value = model.ablate(predicted, inputs, mode)["depth"]
                                        alternatives["fusion-" + mode] = (
                                            value[0, 0].float().cpu().numpy())
                            fused = predicted["depth"][0, 0].float().cpu().numpy()
                    values = {"raw_native": native,
                              "raw_canvas": restore_canvas(
                                  canvas_raw, current.shape, raw["canvas"]),
                              "fusion": restore_canvas(fused, current.shape, raw["canvas"])}
                    if ablations:
                        for name in names[3:]:
                            values[name] = restore_canvas(alternatives.get(name, canvas_raw),
                                                          current.shape, raw["canvas"])
                    for name, value in values.items():
                        if not valid_prediction(value, config):
                            label = f"{dataset}/{sequence['name']}/{i}"
                            raise ValueError(f"Invalid prediction: {label}")
                        arrays[name].append(value if inverse_frontend else 1 / value)
                    if (i + 1) % 50 == 0:
                        print(dataset, sequence["name"], i + 1, "/", len(frames), flush=True)
                record = {"sequence": sequence["name"], "frames": len(frames),
                          "elapsed_inference_s": time.monotonic() - began, "branches": {},
                          "input_stats": difference(runtime, input_before)}
                if inverse_frontend:
                    from explore_a.evaluation.part2_protocol import baseline_sequence_scores
                    record["baseline_fixed_ssi"] = baseline_sequence_scores(
                        arrays, frames, manifest["config"])
                for name, predictions in arrays.items():
                    metrics = official_score(predictions, frames, manifest["config"])
                    from explore_a.evaluation.part2_protocol import aligned_sequence_scores
                    metrics.update(aligned_sequence_scores(
                        predictions, frames, manifest["config"],
                        f"{dataset}:{sequence['name']}"))
                    if dataset == "tum-validation" and not inverse_frontend:
                        unaligned = metric_scores(predictions, frames, manifest["config"])
                        metrics["metric_absrel_secondary"] = unaligned.get("abs_rel")
                        metrics.update({"metric_" + k: v for k, v in unaligned.items()})
                    elif not inverse_frontend:
                        metrics["metric_absrel_secondary"] = metric_absrel(
                            predictions, frames, manifest["config"])
                    record["branches"][name] = metrics
                write_json(receipt, record)
                records.append(record)
                print("COMPLETE", dataset, sequence["name"], record["branches"], flush=True)
            result = {"dataset": dataset, "setting": setting, "checkpoint_step": state["step"],
                      "sequences": records, "frames": sum(r["frames"] for r in records),
                      "mean": {name: {metric: float(np.mean([r["branches"][name][metric]
                                                           for r in records]))
                                      for metric in records[0]["branches"][name]}
                               for name in records[0]["branches"]},
                      "efficiency_note": "Concurrent training; no isolated latency claim"}
            if dataset == "sintel":
                for branch in result["mean"]:
                    result["mean"][branch].update(frame_mean(records, branch))
                result["scoring_protocol"] = manifest["config"]["image_scoring"]
            if inverse_frontend:
                fixed = [r["baseline_fixed_ssi"]["branches"] for r in records]
                result["baseline_fixed_ssi_mean"] = {
                    branch: {key: float(np.mean([r[branch][key] for r in fixed if branch in r]))
                             for key in fixed[0][branch]}
                    for branch in fixed[0]} if fixed and fixed[0] else {}
            write_json(output / dataset / "metrics.json", result)
            results[dataset] = result
    finally:
        runtime.close()
    write_json(output / "summary.json", results)
    return results
