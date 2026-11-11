"""Evaluate registered models by case, sharing frozen inputs and baseline scores."""
import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--registry", type=Path, default=PROJECT/"maintenance/evaluate_models.json")
    parser.add_argument("--output", type=Path, default=PROJECT/"outputs/explore_a/shared_models_evaluation_v2")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--scopes", nargs="+", choices=("dev", "test", "benchmark", "tum-validation"),
                        default=["benchmark", "test", "tum-validation"])
    parser.add_argument("--full-dev", action="store_true")
    parser.add_argument("--setting", choices=("short", "long"), default="long")
    parser.add_argument("--datasets", nargs="+", choices=("kitti", "sintel", "bonn"),
                        default=["kitti", "sintel", "bonn"])
    parser.add_argument("--benchmark-root", type=Path, default=Path.home()/"Documents/ResourcesLow/Datasets")
    parser.add_argument("--tum-root", type=Path, default=Path.home()/"Documents/Resources/Datasets/TUM-RGBD")
    args = parser.parse_args()
    sys.path.insert(0, str(PROJECT/"src"))
    from explore_a.evaluation.session import run, summarize
    if args.command == "run":
        run(args.registry, args.output, args.scopes, args.benchmark_root, args.tum_root,
            args.models, setting=args.setting, datasets=args.datasets, full_dev=args.full_dev)
    else:
        summarize(args.output)


if __name__ == "__main__":
    main()
