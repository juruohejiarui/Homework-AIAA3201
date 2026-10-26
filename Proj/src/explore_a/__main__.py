"""Thin command dispatcher: user starts every preparation, online training and evaluation stage."""
import argparse
import json
from pathlib import Path
import sys

from explore_a.config import DEFAULT_CONFIG, FLOWS, FUSIONS, OCCLUSIONS, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check-weights")
    commands.add_parser("status", help="report configured versus actually prepared datasets")
    plot = commands.add_parser("plot", help="rebuild curves from logs, no model inference")
    plot.add_argument("--run", help="one experiment folder; default: all runs in this config")
    plot.add_argument("--tensorboard", action="store_true", help="also rebuild scalar events")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--extract-kitti", action="store_true")
    evaluate_all = commands.add_parser("evaluate-matrix", help="shared online frontend evaluation")
    evaluate_all.add_argument("--split", choices=("dev", "test"), default="test")
    for action in ("train-occlusion", "train", "evaluate"):
        command = commands.add_parser(action)
        command.add_argument("--flow", choices=FLOWS, default="raft")
        if action != "train-occlusion":
            command.add_argument("--occlusion", choices=OCCLUSIONS, default="rule")
            command.add_argument("--fusion", choices=FUSIONS, default="unet")
        if action == "evaluate":
            command.add_argument("--baseline", choices=("raw", "fixed", "rule-fixed"))
            command.add_argument("--split", choices=("dev", "test"), default="test")
    matrix = commands.add_parser("run-matrix")
    matrix.add_argument("--train-only", action="store_true")
    commands.add_parser("summarize")
    commands.add_parser("all", help="check -> prepare -> online training -> shared evaluation")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "status":
        from explore_a.data.status import dataset_status
        print(json.dumps(dataset_status(config), indent=2, ensure_ascii=False))
    elif args.command == "plot":
        from explore_a.observability.export import export
        export(config, args.run, args.tensorboard)
    elif args.command == "check-weights":
        from explore_a.models.weights import check_weights
        check_weights(config)
    elif args.command == "prepare":
        from explore_a.data.prepare import prepare
        prepare(config, args.extract_kitti)
    elif args.command == "evaluate-matrix":
        from explore_a.evaluation.engine import evaluate_matrix
        evaluate_matrix(config, args.split)
    elif args.command in ("train", "train-occlusion"):
        from explore_a.training.engine import run_training
        run_training(config, args.flow, getattr(args, "occlusion", "rule"),
                     getattr(args, "fusion", "unet"), args.command == "train-occlusion")
    elif args.command == "evaluate":
        from explore_a.evaluation.engine import evaluate
        evaluate(config, args.flow, args.occlusion, args.fusion, args.baseline, args.split)
    elif args.command == "summarize":
        from explore_a.matrix import summarize
        summarize(config)
    elif args.command == "run-matrix":
        from explore_a.matrix import run_matrix
        run_matrix(config, not args.train_only)
    elif args.command == "all":
        from explore_a.data.prepare import load_manifest
        from explore_a.matrix import command, run_matrix
        command(config, "check-weights")
        if not Path(config["paths"]["manifest"]).exists():
            command(config, "prepare", "--extract-kitti")
        else:
            load_manifest(config)
        run_matrix(config)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
