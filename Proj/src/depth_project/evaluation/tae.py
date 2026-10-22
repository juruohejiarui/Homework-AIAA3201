"""Run original ScanNet bidirectional 3D TAE, separately from optical-flow MAE."""
import argparse
from pathlib import Path

from depth_project.evaluation.video import evaluate_sequences
from depth_project.io import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_sequences(read_json(args.manifest), "tae", args.predictions, tae=True)
    write_json(args.output, result)
    print(result["mean"])


if __name__ == "__main__":
    main()
