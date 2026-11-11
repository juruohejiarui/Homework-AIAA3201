"""Four-model comparison CLI; separate processes release GPU memory between references."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from depth_project.io import write_json
from depth_project.paths import DATASETS
from explore_a.config import file_identity, load_config
from explore_a.data.prepare import load_manifest
from explore_a.training.dev_subset import select_dev

MODELS = ("fusion", "unidepthv2", "da2")


def scope_root(args):
    root = args.output / args.scope
    if args.scope == "benchmark":
        return root / ("vda-" + args.setting)
    if args.scope == "tum-validation":
        return root / "full-official-validation"
    return root


def scope_counts(args, datasets):
    from depth_project.data.protocol import select_sequences
    from explore_a.evaluation.part2 import MANIFESTS, benchmark_root_for

    counts = {}
    for dataset in datasets:
        root = benchmark_root_for(dataset, args.benchmark_root, args.tum_root)
        manifest = json.loads((root / MANIFESTS[dataset]).read_text())
        selected = select_sequences(manifest, args.setting)
        counts[dataset] = {"sequences": len(selected),
                           "frames": sum(len(s["frames"]) for s in selected),
                           "per_sequence": {s["name"]: len(s["frames"]) for s in selected}}
    return counts


def cohort(config, split, full_dev=False):
    rows = [r for r in load_manifest(config)["samples"] if r["split"] == split]
    if split == "dev" and not full_dev:
        rows = select_dev(rows, config["training"]["dev_sample_fraction"],
                          config.get("data", {}).get("seed", config["training"]["seed"]))
    if not rows:
        raise ValueError("Empty comparison cohort")
    return rows


def pin(config, checkpoint, output):
    """Pin one small head snapshot across every scope, without copying optimizer state."""
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "run.json"
    identity = {"checkpoint_source": str(checkpoint.resolve()),
                "manifest": file_identity(config["paths"]["manifest"]),
                "data": config["data"], "geometry": config["geometry"],
                "reference_protocol": "v1; native RGB -> same canvas for dev/test",
                "dev_fraction": config["training"]["dev_sample_fraction"],
                "dev_seed": config["training"]["seed"]}
    if receipt.exists() and json.loads(receipt.read_text()) != identity:
        raise ValueError("Output directory belongs to a different comparison")
    snapshot = output / "checkpoint.pt"
    if not snapshot.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = {k: state[k] for k in ("model", "identity", "step", "best")}
        pending = snapshot.with_suffix(".pt.part")
        torch.save(state, pending)
        pending.replace(snapshot)
    write_json(receipt, identity)
    return snapshot


def run_split(config, checkpoint, output, split, model, full_dev):
    rows = cohort(config, split, full_dev)
    root = output / ("dev-full" if split == "dev" and full_dev else split)
    from explore_a.training.dev_subset import selection_record
    ids = [r["id"] for r in rows]
    write_json(root / "cohort.json", {
        **selection_record(rows), "split": split, "sample_ids": ids, "pairs": len(rows),
        "sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "dev_selection": "full" if full_dev else "same fixed subset as training"})
    if model != "fusion":
        from explore_a.evaluation.reference_splits import run
        return run(config, rows, model, root / model)
    from explore_a.evaluation.branches import specification
    from explore_a.evaluation.engine import evaluate_specs

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    task = state["identity"]["experiment"]
    spec = specification(**task)
    spec["checkpoint"] = checkpoint
    baseline = specification(task["flow"], baseline="raw")
    # Evaluation shares the configured frozen pipeline; consumers never write predictions.
    config = copy.deepcopy(config)
    return evaluate_specs(config, [spec, baseline], split, output=root / "fusion", rows=rows)


def run_scope(args, config, checkpoint):
    root = scope_root(args)
    if args.scope in ("dev", "test"):
        return run_split(config, checkpoint, args.output, args.scope, args.model, args.full_dev)
    if args.scope in ("benchmark", "tum-validation"):
        datasets = ["tum-validation"] if args.scope == "tum-validation" else args.datasets
        if args.model == "fusion":
            from explore_a.evaluation.part2 import evaluate_part2
            return evaluate_part2(config, checkpoint, root / "fusion", args.benchmark_root,
                                  args.setting, datasets, tum_root=args.tum_root)
        from explore_a.evaluation.part2_references import run
        return run(args.model, root / args.model, args.benchmark_root, datasets,
                   args.setting, tum_root=args.tum_root)
    from explore_a.evaluation.comparison_nyu import run
    return run(config, args.nyu_root, checkpoint, root, args.model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope",
                        choices=("dev", "test", "benchmark", "nyu", "tum-validation"),
                        required=True)
    parser.add_argument("--model", choices=("all", *MODELS), default="all")
    parser.add_argument("--full-dev", action="store_true")
    parser.add_argument("--benchmark-root", type=Path, default=DATASETS)
    parser.add_argument("--tum-root", type=Path,
                        default=Path.home() / "Documents/Resources/Datasets/TUM-RGBD")
    parser.add_argument("--nyu-root", type=Path, default=DATASETS / "nyu_depth_v2")
    parser.add_argument("--setting", choices=("short", "long", "full"),
                        help="default: VDA main-table long; TUM all matched validation frames")
    parser.add_argument("--datasets", nargs="+", choices=("kitti", "sintel", "bonn"),
                        default=["kitti", "sintel", "bonn"])
    args = parser.parse_args()
    if args.setting is None:
        args.setting = "full" if args.scope == "tum-validation" else "long"
    if args.scope == "tum-validation" and args.setting != "full":
        parser.error("TUM has no published short/long frame cap; use --setting full")
    if args.scope == "benchmark" and args.setting == "full":
        parser.error("VDA main table uses --setting long (up to 500 frames), not full")
    if args.scope in ("dev", "test", "benchmark", "tum-validation"):
        from explore_a.evaluation.session import run as run_session
        from depth_project.inference.__main__ import WEIGHTS
        from depth_project.paths import CHECKPOINTS
        models = []
        if args.model in ("all", "fusion"):
            models.append({"name": "fusion", "config": str(args.config.resolve()),
                           "checkpoint": str(args.checkpoint.resolve())})
        if args.model in ("all", "da2"):
            models.append({"name": "da2", "kind": "frozen", "adapter": "da2",
                           "config": str(args.config.resolve()), "weights": str(CHECKPOINTS/WEIGHTS["da2"]),
                           "reuse_frontend": True, "load_param": {"fp32": False}})
        if args.model in ("all", "unidepthv2"):
            models.append({"name": "unidepthv2", "kind": "frozen", "adapter": "unidepthv2",
                           "config": str(args.config.resolve()), "weights": str(CHECKPOINTS/WEIGHTS["unidepthv2"]),
                           "load_param": {"fp32": True}})
        args.output.mkdir(parents=True, exist_ok=True)
        registry = args.output/"requested_models.json"
        write_json(registry, models)
        run_session(registry, args.output, [args.scope], args.benchmark_root, args.tum_root,
                    setting="long" if args.scope == "tum-validation" else args.setting,
                    datasets=args.datasets, full_dev=args.full_dev)
        return
    config = load_config(args.config)
    checkpoint = pin(config, args.checkpoint, args.output)
    # Keep short/long receipts separate; a changed setting must not reuse earlier metrics.
    if args.scope in ("benchmark", "tum-validation"):
        path = scope_root(args) / "protocol.json"
        datasets = ["tum-validation"] if args.scope == "tum-validation" else args.datasets
        identity = {"setting": args.setting, "datasets": datasets,
                    "root": str(args.benchmark_root.resolve()),
                    "cohort_counts": scope_counts(args, datasets),
                    "paper_scope": ("VDA main-table length" if args.setting == "long"
                                    else "VDA appendix length" if args.setting == "short"
                                    else "all official TUM validation; paper list unavailable"),
                    "exact_paper_numeric_reproduction": False,
                    "note": "Sintel meter GT and Bonn timestamp association are documented "
                            "decoding fixes; model sizes/precision follow this comparison"}
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError("Benchmark setting changed; use a separate comparison output")
        write_json(path, identity)
        print("EVALUATION PROTOCOL", args.scope, args.setting, "datasets", datasets,
              "output", scope_root(args), flush=True)
        print("FRAME COUNTS", {k: v["frames"] for k, v in identity["cohort_counts"].items()},
              flush=True)
    if args.model == "all":
        for model in MODELS:
            command = [sys.executable, "-m", "explore_a.evaluation.comparison", *sys.argv[1:]]
            # argparse accepts the last --model occurrence; all children use the same snapshot.
            subprocess.run([*command, "--model", model], check=True)
        from explore_a.evaluation.comparison_report import report
        report(args.output)
    else:
        run_scope(args, config, checkpoint)


if __name__ == "__main__":
    main()
