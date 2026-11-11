"""Online single/multi-head evaluation with shared frontends and bounded result writing."""
from contextlib import closing
import copy
from pathlib import Path
import time

import numpy as np
import torch

from depth_project.io import write_json
from explore_a.config import experiment_name
from explore_a.data.prepare import load_manifest
from explore_a.evaluation.accumulator import EvaluationAccumulator
from explore_a.evaluation.branches import load_branch, matrix_specs, specification
from explore_a.evaluation.output import ResultSink
from explore_a.geometry.warp import pair_geometry
from explore_a.online.factory import make_runtime
from explore_a.online.frozen import synchronize
from explore_a.observability.progress import Progress, message
from explore_a.online.loading import OnlinePairs
from explore_a.online.statistics import snapshot
from explore_a.training.prediction import predict, project_to_interval, global_only as only_global
from explore_a.models.inputs import FusionInputs


def baseline_output(name, data, geometry):
    weight = torch.zeros_like(data["current_depth"])
    if name != "raw":
        mask = geometry["rule"] if name == "rule-fixed" else geometry["hard"]
        weight = mask.float() * .5
    return {"depth": (1 - weight) * data["current_depth"] + weight * geometry["historical"],
            "historical_weight": weight, "visibility": geometry["hard"].float(),
            "prediction_kind": geometry.get("prediction_kind", "metric_m")}


def evaluate(config, flow="raft", occlusion="rule", fusion="unet_guided", baseline=None,
             split="test", device="cuda", project_interval=False, global_only=False,
             checkpoint=None, output=None, ablation="full"):
    spec = specification(flow, occlusion, fusion, baseline, project_interval, global_only, ablation)
    if checkpoint is not None:
        spec["checkpoint"] = checkpoint
    return evaluate_specs(config, [spec], split, device, output)[0]


def evaluate_matrix(config, split="test", device="cuda"):
    return evaluate_specs(config, matrix_specs(), split, device)


@torch.inference_mode()
def evaluate_specs(config, specs, split="test", device="cuda", output=None, rows=None):
    if output is not None:
        root = Path(config["paths"]["runs"])
        specs = [{**spec, "checkpoint": spec.get("checkpoint") or
                  root / experiment_name(spec["flow"], spec["occlusion"], spec["fusion"]) /
                  "best.pt"} if not spec["baseline"] else spec for spec in specs]
        config = copy.deepcopy(config)
        config["paths"]["runs"] = str(output)
    manifest = load_manifest(config)
    if rows is None:
        rows = [row for row in manifest["samples"] if row["split"] == split]
        if split == "dev":
            from explore_a.training.dev_subset import select_dev
            rows = select_dev(rows, config["training"]["dev_sample_fraction"],
                              config["data"]["seed"])
    if any(row["split"] != split for row in rows):
        raise ValueError("Explicit evaluation rows contain a different split")
    if not rows:
        raise ValueError(f"No evaluation samples in {split}")
    branches, models = [], {}
    for spec in specs:
        key = (spec["flow"], spec["occlusion"], spec["fusion"], str(spec.get("checkpoint")))
        if not spec["baseline"] and key in models:
            branch = {**spec, **{name: models[key][name]
                                for name in ("model", "checkpoint", "checkpoint_step")}}
        else:
            branch = load_branch(config, spec, device)
            if not spec["baseline"]:
                models[key] = branch
        branches.append(branch)
    # Farneback is the fixed, common temporal-scoring reference for every experiment.
    flows = sorted({branch["flow"] for branch in branches} | {"farneback"})
    teacher_flows = {b["flow"] for b in branches
                     if b["occlusion"] == "mftiq" and not b["baseline"]}
    runtime = make_runtime(config, flows, bool(teacher_flows), device)
    dataset = OnlinePairs(config, rows, runtime, flows[0], phase=split)
    shared = len(branches) > 1
    if shared:
        (Path(config["paths"]["runs"]) / f"matrix-{split}.json").unlink(missing_ok=True)
    front = {name: [] for name in ("depth", *flows)}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    with closing(runtime), ResultSink(config["result_io"]) as sink:
        accumulators = [EvaluationAccumulator(config, b["name"], split, sink) for b in branches]
        label = (f"EVAL-MATRIX {split} ({len(branches)} branches)" if shared
                 else f"EVAL {split} {branches[0]['name']}")
        with (closing(dataset.raw_stream(flows=flows, teacher_flows=teacher_flows)) as loaded,
              Progress(config, label, len(rows)) as progress):
            iterator = iter(loaded)
            for _ in rows:
                pair_began = time.perf_counter()
                index, raw = next(iterator)
                row = rows[index]
                progress.set(dataset=row["dataset"], sequence=row["sequence"], phase="frontends")
                base, elapsed = runtime.timed_depth(raw)
                front["depth"].append(elapsed)
                inputs = {}
                for flow in flows:
                    synchronize(device)
                    started = time.perf_counter()
                    data = runtime.add_flow(raw, base, flow, flow in teacher_flows)
                    inputs[flow] = {key: value.unsqueeze(0) for key, value in data.items()}
                    synchronize(device)
                    front[flow].append(time.perf_counter() - started)
                predictions = {}
                for branch, accumulator in zip(branches, accumulators):
                    data = inputs[branch["flow"]]
                    progress.set(branch=branch["name"], phase="heads")
                    synchronize(device)
                    started = time.perf_counter()
                    with torch.autocast(device_type=device, enabled=device == "cuda"
                                        and config["training"]["amp"]):
                        if branch["baseline"]:
                            from explore_a.training.prediction import fusion_geometry
                            output = baseline_output(branch["baseline"], data,
                                                     fusion_geometry(data, config))
                        else:
                            key = id(branch["model"])
                            if key in predictions:
                                output, geometry = predictions[key]
                            else:
                                output, geometry = predict(branch["model"], data, config,
                                                           branch["occlusion"])
                                predictions[key] = output, geometry
                            if branch.get("ablation", "full") != "full":
                                visibility = (torch.ones_like(data["current_depth"])
                                              if branch["occlusion"] == "rule"
                                              else data["teacher_visibility"])
                                fusion_inputs = FusionInputs.from_pair(data, geometry, visibility)
                                output = branch["model"].ablate(output, fusion_inputs,
                                                               branch["ablation"])
                            if branch.get("project_interval"):
                                output = project_to_interval(output, data)
                            if branch.get("global_only"):
                                output = only_global(output, data)
                    synchronize(device)
                    accumulator.head_timings.append(time.perf_counter() - started)
                    if not shared:
                        accumulator.timings.append(time.perf_counter() - pair_began)
                    accumulator.add(row, data, output, inputs["farneback"])
                stats = snapshot(runtime)
                if stats:
                    progress.set(cache_hits=sum(v for k, v in stats.items() if k.endswith("_hit")),
                                 online=sum(v for k, v in stats.items() if k.endswith("_online")))
                progress.update(1)
        message("Saving evaluation results: draining ordered metrics and image writers")
    # All JSONL and image futures have succeeded before any final metrics summary is published.
    wall = time.perf_counter() - began
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
    results = []
    for branch, accumulator in zip(branches, accumulators):
        times, heads = accumulator.timings, accumulator.head_timings
        cached_mode = config.get("execution", {}).get("backend") == "cache"
        latency = {"mode": ("shared_cached" if shared else "independent_cached")
                   if config.get("execution", {}).get("backend") == "cache" else
                   ("shared_frontend" if shared else "independent_online"),
                   "scope": ("local cache reads/transfer/fusion; excludes all DGX generation"
                             if cached_mode
                             else "prefetch wait through prediction, including frozen frontends "
                             "and fixed-Farneback preparation; excludes scoring/output/model load"),
                   "mean_ms": float(np.mean(times) * 1000) if times else None,
                   "warm_ms": float(np.mean(times[5:] or times) * 1000) if times else None,
                   "fusion_mean_ms": float(np.mean(heads) * 1000),
                   "fusion_warm_ms": float(np.mean(heads[5:] or heads) * 1000),
                   "frontend_mean_ms": {key: float(np.mean(values) * 1000)
                                        for key, values in front.items()},
                   "frontend_note": "Cache mode measures reads, not DGX generation; flow timing "
                                    "includes MFTIQ only for teacher-required flows",
                   "backend": config.get("execution", {}).get("backend", "online"),
                   "pipeline_wall_s": wall, "pairs_per_second": len(rows) / max(wall, 1e-9),
                   "memory_scope": "entire process, all resident heads/frontends and scoring",
                   "peak_allocated_gib": peak}
        if config.get("execution", {}).get("backend") == "hybrid":
            latency.update(mode="shared_hybrid" if shared else "independent_hybrid",
                           scope="blocking cache/DGX/local components + fusion; "
                                 "includes lazy loads, excludes prior cache generation",
                           input_stats=snapshot(runtime),
                           frontend_note="Hybrid online timings include lazy load time; "
                                         "model_load_s is reported separately, not added twice. "
                                         "Counts are shared frontend totals in matrix evaluation.")
        result = accumulator.finish(manifest, latency)
        result["projection_ablation"] = bool(branch.get("project_interval"))
        result["global_only_ablation"] = bool(branch.get("global_only"))
        result["ablation"] = branch.get("ablation", "full")
        if branch.get("checkpoint"):
            result.update(checkpoint=str(branch["checkpoint"]),
                          checkpoint_step=branch["checkpoint_step"])
        if branch["fusion"] == "final_flow":
            result["ablation_timing_note"] = ("Shared normal forward may be reused; ablation "
                                               "head timing is not standalone model latency")
        if branch["model"] is not None:
            parameters = list(branch["model"].parameters())
            result["fusion_model"] = {
                "architecture": branch["fusion"], "parameters": sum(p.numel() for p in parameters),
                "parameter_bytes": sum(p.numel() * p.element_size() for p in parameters),
                "scope": "fusion head only; frozen frontends excluded",
            }
        write_json(accumulator.root / "metrics.json", result)
        message(accumulator.name, result["by_dataset"])
        results.append(result)
    if shared:
        write_json(Path(config["paths"]["runs"]) / f"matrix-{split}.json", {
            "experiments": [result["experiment"] for result in results],
            "pairs": len(rows), "wall_seconds": wall,
            "note": "Shared frontend throughput; per-head latency is not isolated end-to-end",
        })
    return results
