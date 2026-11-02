"""Positive control: test-room leakage is detected, while unused official val is harmless."""
import h5py
import numpy as np
import pytest
from scipy.io import savemat
import torch

from depth_project.io import write_json
from explore_a.audits.nyu_leakage import run
from explore_a.config import file_identity


@pytest.mark.parametrize("leak", [False, True])
def test_official_room_leakage_positive_control(tmp_path, leak):
    h5root = tmp_path / "h5"
    official = h5root / "val/official/00002.h5"
    official.parent.mkdir(parents=True)
    with h5py.File(official, "w") as f:
        f["rgb"] = np.ones((3, 4, 6), np.uint8)
    with h5py.File(tmp_path / "nyu_depth_v2_labeled.mat", "w") as f:
        a = f.create_dataset("room_a", data=np.array([ord(c) for c in "office_0001a"]))
        b = f.create_dataset("room_b", data=np.array([ord(c) for c in "office_0002b"]))
        refs = f.create_dataset("scenes", (2, 1), dtype=h5py.ref_dtype)
        refs[0, 0], refs[1, 0] = a.ref, b.ref
        f["images"] = np.ones((2, 3, 6, 4), np.uint8)
    savemat(tmp_path / "splits.mat", {"trainNdxs": [[1]], "testNdxs": [[2]]})
    scene = "office_0002a" if leak else "office_0001b"
    row = {"dataset": "nyuv2", "split": "train", "sequence": scene, "id": "pair",
           "previous_id": "00001", "current_id": "00006",
           "previous_rgb": str(h5root / "train" / scene / "00001.h5"),
           "current_rgb": str(h5root / "train" / scene / "00006.h5")}
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"samples": [row]})
    checkpoint = tmp_path / "best.pt"
    torch.save({"step": 1, "identity": {"input": {"manifest": file_identity(manifest)}}},
               checkpoint)
    config = {"paths": {"manifest": str(manifest)}, "data": {"nyu_h5_root": str(h5root)}}
    result = run(config, tmp_path, checkpoint, tmp_path / "audit.json")
    assert result["training_or_selection_room_leakage_detected"] == leak
    assert result["checkpoint_manifest_identity_matches"]
    assert result["manifest_uses_val_frames"] == []
    assert result["h5_val_rgb_exact_matches"] == 1
