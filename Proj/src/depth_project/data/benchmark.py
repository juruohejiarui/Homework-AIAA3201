"""Create a reusable, explicit RGB/GT sequence manifest."""
import argparse
from pathlib import Path

from depth_project.data.bonn import prepare_bonn
from depth_project.data.exported import prepare_exported, prepare_kitti
from depth_project.data.protocol import CONFIG
from depth_project.data.sintel import prepare_sintel
from depth_project.io import write_json
from depth_project.paths import DATASETS
from depth_project.storage.mount import require_dataset_volume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=CONFIG, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--depth-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    require_dataset_volume()
    config = CONFIG[args.dataset]
    base = DATASETS / config["folder"]
    source = args.source or base / "raw"
    output = args.output or base / "prepared/vda_protocol_v2/manifest.json"
    if args.dataset == "sintel":
        sequences = prepare_sintel(source, output.parent)
    elif args.dataset == "bonn":
        sequences = prepare_bonn(source)
    elif args.dataset == "kitti":
        if args.depth_root is None:
            parser.error("KITTI requires --depth-root")
        sequences = prepare_kitti(source, args.depth_root)
    else:
        sequences = prepare_exported(source, args.dataset)
    if not sequences:
        raise FileNotFoundError(f"No sequences found in {source}")
    write_json(output, {"version": 2, "dataset": args.dataset, "config": config,
                        "sequences": sequences, "protocol": "Official code selection with documented preprocessing corrections; paper cohort unconfirmed"})
    print(output)


if __name__ == "__main__":
    main()
