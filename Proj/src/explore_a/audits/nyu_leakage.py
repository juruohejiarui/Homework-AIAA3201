"""Audit official labeled test-room isolation without decoding the H5 training collection."""
import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat
import torch

from depth_project.io import read_json, write_json
from explore_a.config import file_identity, load_config
from explore_a.data.nyu import room_id
from explore_a.data.nyu_h5 import read_h5


def temporal_metadata(source, test_ids):
    """Labeled images retain source timestamps, but are sparse video samples."""
    if "rawRgbFilenames" not in source:
        return {"available": False}
    refs = source["rawRgbFilenames"][:].ravel()
    groups = defaultdict(list)
    for index in test_ids:
        raw = "".join(chr(int(c)) for c in source[refs[index - 1]][:].ravel())
        path = Path(raw)
        groups[str(path.parent)].append(float(path.name.split("-")[1]))
    gaps = [b - a for rows in groups.values()
            for a, b in zip(sorted(rows), sorted(rows)[1:])]
    quantiles = dict(zip(("min", "p25", "median", "p75", "max"),
                         map(float, np.quantile(gaps, [0, .25, .5, .75, 1])))) if gaps else {}
    return {"available": True, "test_images": len(test_ids), "raw_sequences": len(groups),
            "sequences_with_multiple_images": sum(len(v) > 1 for v in groups.values()),
            "adjacent_labeled_pairs_within_same_raw_sequence": len(gaps),
            "seconds_quantiles": quantiles,
            "pairs_gap_at_most_0_1s": sum(g <= .1 for g in gaps),
            "pairs_gap_at_most_1s": sum(g <= 1 for g in gaps)}


def run(config, root, checkpoint, output):
    root, checkpoint = Path(root), Path(checkpoint)
    split = loadmat(root / "splits.mat")
    test_ids = split["testNdxs"].ravel().astype(int)
    manifest = read_json(config["paths"]["manifest"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    rows = [r for r in manifest["samples"] if r["dataset"] == "nyuv2"]
    groups, sequences, frames, pair_counts = (defaultdict(set) for _ in range(4))
    val_paths, malformed = [], []
    for row in rows:
        part = row["split"]
        groups[part].add(room_id(row["sequence"]))
        sequences[part].add(row["sequence"])
        pair_counts[part].add(row["id"])
        for prefix in ("previous", "current"):
            path = Path(row[prefix + "_rgb"])
            frames[part].add(str(path))
            if "/h5/val/" in str(path):
                val_paths.append(str(path))
            if (path.parent.name != row["sequence"] or path.stem != row[prefix + "_id"]
                    or path.parent.parent.name != "train"):
                malformed.append(str(path))
    with h5py.File(root / "nyu_depth_v2_labeled.mat", "r") as source:
        refs = source["scenes"][:].ravel()
        names = ["".join(chr(int(c)) for c in source[ref][:].ravel()) for ref in refs]
        official_train = {room_id(names[int(i) - 1]) for i in split["trainNdxs"].ravel()}
        official_test = {room_id(names[int(i) - 1]) for i in test_ids}
        temporal = temporal_metadata(source, test_ids)
        files = sorted((Path(config["data"]["nyu_h5_root"]) / "val").rglob("*.h5"),
                       key=lambda p: int(p.stem))
        matched, mismatches = 0, []
        if len(files) != len(test_ids):
            raise ValueError("H5 val count differs from the official labeled test count")
        ids_match = all(int(path.stem) == int(i) for path, i in zip(files, test_ids))
        if not ids_match:
            raise ValueError("H5 val filenames differ from the official labeled test IDs")
        for index, path in enumerate(files):
            rgb, _ = read_h5(path, depth=False)
            expected = source["images"][test_ids[index] - 1].transpose(2, 1, 0)
            if np.array_equal(rgb, expected):
                matched += 1
            else:
                error = np.abs(rgb.astype(np.float32) - expected.astype(np.float32))
                changed = (error > 0).any(2)
                mismatches.append({"h5": str(path), "mat_id": int(test_ids[index]),
                                   "mean_abs_rgb_0_255": float(error.mean()),
                                   "changed_pixels": int(changed.sum()),
                                   "changed_pixel_fraction": float(changed.mean())})
            if (index + 1) % 100 == 0:
                print("OFFICIAL H5 VAL", index + 1, "/", len(files), flush=True)
    details = {}
    for part in ("train", "dev", "test"):
        details[part] = {"pairs": len(pair_counts[part]), "sequences": len(sequences[part]),
                         "rooms": len(groups[part]), "unique_frames": len(frames[part]),
                         "official_test_room_overlap": sorted(groups[part] & official_test),
                         "outside_official_train_rooms": sorted(groups[part] - official_train)}
    leaked = {part: d["official_test_room_overlap"] for part, d in details.items()
              if part in ("train", "dev") and d["official_test_room_overlap"]}
    saved = state["identity"]["input"]["manifest"]
    current = file_identity(config["paths"]["manifest"])
    result = {"checkpoint_step": state["step"], "checkpoint": str(checkpoint.resolve()),
              "official_labeled_train_images": len(split["trainNdxs"]),
              "official_labeled_test_images": len(test_ids),
              "official_train_rooms": len(official_train),
              "official_test_rooms": len(official_test),
              "official_train_test_shared_rooms": sorted(official_train & official_test),
              "official_labeled_temporal_metadata": temporal,
              "h5_val_file_ids_match_official_test_ids": ids_match,
              "manifest_splits": details, "h5_val_rgb_exact_matches": matched,
              "h5_val_mismatches": mismatches, "manifest_uses_val_frames": sorted(set(val_paths)),
              "malformed_h5_frame_identities": sorted(set(malformed)),
              "saved_manifest_identity": saved, "current_manifest_identity": current,
              "checkpoint_manifest_identity_matches": saved == current,
              "train_dev_test_frame_overlap": {
                  a + "/" + b: len(frames[a] & frames[b])
                  for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))},
              "training_or_selection_room_leakage_detected": bool(leaked or val_paths or malformed),
              "scope": "All manifest endpoints and labeled room IDs; all 654 H5 val RGBs. "
                       "No full H5 training-image hashing or foundation pretraining audit."}
    write_json(output, result)
    summary = {"room_overlap": leaked, "val_used": len(val_paths),
               "official_val_exact": matched, "manifest_matches": saved == current}
    print("NYU LEAKAGE AUDIT", summary, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(load_config(args.config), args.root, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
