"""H5 metric reading, retained-frame isolation, shared identities and RGB-only transport."""
import io
from pathlib import Path
import tarfile

import h5py
import numpy as np
import pytest
import torch

from depth_project.io import read_json, write_json
from explore_a.cache import plan, transfers
from explore_a.cache.identity import assert_compatible
from explore_a.cache.migration import migrate
from explore_a.cache.paths import artifact, expected
from explore_a.cache.policy import count_rgb
from explore_a.cache.producer import Producer, decoded
from explore_a.cache.inputs import RGBInputs
from explore_a.online.factory import make_runtime
from explore_a.cache.settings import root_path
from explore_a.data.nyu_h5 import extract, read_h5
from explore_a.data.nyu_h5_pairs import prepare_h5
from explore_a.online.loading import read_pair
from explore_a.training.sampling import StepBatches
from test_explore_a_distributed import prepared as prepared, fake_transport
from test_explore_a_pipeline import fake_backbones as fake_backbones


def h5_file(path, value=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.broadcast_to(np.array([10, 40, 90], np.uint8)[:, None, None], (3, 16, 24))
    with h5py.File(path, "w") as f:
        f["rgb"] = rgb
        f["depth"] = np.ones((16, 24), np.float32) * value


def h5_row(tmp_path, sequence="room", split="train"):
    folder = tmp_path / sequence
    for name in ("00001", "00006"):
        h5_file(folder / (name + ".h5"))
    return {"id": f"nyuv2:{sequence}:h5_rgb:00006", "dataset": "nyuv2",
            "sequence": sequence, "camera": "h5_rgb", "group": sequence, "split": split,
            "previous_id": "00001", "current_id": "00006",
            "previous_rgb": str(folder / "00001.h5"),
            "current_rgb": str(folder / "00006.h5"), "gt": str(folder / "00006.h5"),
            "previous_gt": str(folder / "00001.h5"), "depth_encoding": "nyu_h5",
            "K": None, "pose": None}


def test_safe_resumable_extraction_and_unclaimed_conflict(tmp_path):
    archive = tmp_path / "archives"
    archive.mkdir()
    h5_file(tmp_path / "original.h5")
    payload = (tmp_path / "original.h5").read_bytes()
    with tarfile.open(archive / "train-000000.tar", "w") as t:
        m = tarfile.TarInfo("train/room/00001.h5")
        m.size = len(payload)
        t.addfile(m, io.BytesIO(payload))
    output = tmp_path / "h5"
    assert extract(archive, output) == {"train": 1}
    target = output / "train/room/00001.h5"
    stamp = target.stat().st_mtime_ns
    assert extract(archive, output) == {"train": 1}
    assert target.stat().st_mtime_ns == stamp
    unknown = tmp_path / "unknown/train/room/00001.h5"
    unknown.parent.mkdir(parents=True)
    unknown.write_bytes(b"x" * len(payload))
    with pytest.raises(ValueError, match="Conflicting"):
        extract(archive, tmp_path / "unknown")
    bad = tmp_path / "bad"
    bad.mkdir()
    with tarfile.open(bad / "train-000000.tar", "w") as t:
        m = tarfile.TarInfo("../escape.h5")
        t.addfile(m, io.BytesIO())
    with pytest.raises(ValueError, match="Unsafe"):
        extract(bad, tmp_path / "unsafe")


def test_h5_room_isolation_numeric_order_and_metric_depth(tmp_path, monkeypatch):
    from explore_a.data import nyu_h5_pairs
    scenes = ("bedroom_0001a", "bedroom_0001b", "office_0002", "office_0003", "office_0004")
    for name in scenes:
        for number in (1, 6, 11):
            h5_file(tmp_path / "h5/train" / name / f"{number:05d}.h5")
    h5_file(tmp_path / "h5/val/official/00001.h5")
    monkeypatch.setattr(nyu_h5_pairs, "test_rooms", lambda _: {"bedroom_0001"})
    rows, report = prepare_h5(tmp_path, {"nyu_h5_root": str(tmp_path / "h5"),
                                        "seed": 2026, "dev_fraction": .15}, {})
    assert len(rows) == 10 and report["official_validation_unpaired"] == 1
    assert all(r["split"] == "test" for r in rows if r["sequence"].startswith("bedroom"))
    assert all(r["frame_index_gap"] == 5 and r["time_gap_s"] is None for r in rows)
    assert not any("official" in r["current_rgb"] for r in rows)
    for row in rows:
        assert int(row["previous_id"]) < int(row["current_id"])
    rgb, depth = read_h5(rows[0]["current_rgb"])
    np.testing.assert_array_equal(rgb[0, 0], [10, 40, 90])
    assert depth.dtype == np.float32 and depth.shape == (16, 24)
    np.testing.assert_array_equal(depth, 3)
    from explore_a.data.splits import validate_rows
    validate_rows(rows, {})


def test_weighted_batches_resume_and_homogeneous_microbatches():
    rows = [{"dataset": name} for name in ("kitti", "tum", "nyuv2")]
    weights = {"kitti": .5, "tum": .25, "nyuv2": .25}
    all_batches = list(StepBatches(rows, 8, 4, 10000, 0, 2026, weights))
    resumed = list(StepBatches(rows, 8, 4, 10000, 9000, 2026, weights))
    assert resumed == all_batches[18000:]
    counts = {name: 0 for name in weights}
    for first, second in zip(all_batches[::2], all_batches[1::2]):
        assert len(set(first + second)) == 1
        counts[rows[first[0]]["dataset"]] += 1
    assert all(abs(counts[k] / 10000 - weights[k]) < .02 for k in weights)
    with pytest.raises(ValueError):
        StepBatches(rows, 8, 4, 1, 0, 2026, {"kitti": 1})


def test_shared_recipe_adds_h5_and_rgb_transport_matches_local(prepared, tmp_path, monkeypatch):
    config, base, root, name = prepared
    before_job = (root / "jobs" / name / "job.json").read_bytes()
    # An existing prediction must survive extension and is never recomputed by planning.
    original_row = read_json(config["paths"]["manifest"])["samples"][0]
    existing = artifact(root, "frames", original_row)
    from explore_a.cache.storage import write_npz
    write_npz(existing, depth=np.ones((1, 16, 24), np.float32))
    stamp = existing.stat().st_mtime_ns
    old_spec = read_json(root / "recipe.json")
    config["data"]["datasets"].append("nyuv2")
    manifest = read_json(config["paths"]["manifest"])
    manifest["settings"] = config["data"]
    row = h5_row(tmp_path / "h5")
    manifest["samples"].append(row)
    write_json(config["paths"]["manifest"], manifest)
    plan.prepare_plan(config, base=base)
    assert_compatible(old_spec, read_json(root / "recipe.json"))
    assert existing.stat().st_mtime_ns == stamp
    assert (root / "jobs" / name / "job.json").read_bytes() == before_job
    job = plan.load_job(root, "nyuv2_h5__room")
    sources = read_json(root / "jobs/nyuv2_h5__room/sources.json")
    assert count_rgb(job, sources) == 2
    local = decoded(job["rows"][0], RGBInputs(job, sources=sources), config)
    remote_base = tmp_path / "dgx"
    fake_transport(monkeypatch, remote_base)
    transfers.upload_controls(root)
    transfers.upload(root, job["job"])
    target = root_path(remote_base)
    remote_job = plan.load_job(target, job["job"])
    assert remote_job["inputs"] != job["inputs"]
    staged = target / ".staging" / job["job"]
    assert not list(staged.rglob("*.h5"))
    assert not list(staged.rglob("*.rgb.json"))
    raw = decoded(remote_job["rows"][0], RGBInputs(remote_job, staging=staged), config)
    np.testing.assert_array_equal(local["images"][False], raw["images"][False])
    assert remote_job["outputs"] == expected([row], frames=False)
    Producer(config, "cpu").generate(target, remote_job, staged, 0)
    transfers.pull(root, job["job"], 0)
    config["cache"] = {"root": str(base), "recipe": root.name}
    runtime = make_runtime(config, ["raft"], True, "cpu")
    data = runtime.compute(read_pair(row, config["data"]), "raft", True)
    assert torch.all(data["gt"] == 3) and torch.all(data["current_depth"] == 2)
    layer = torch.nn.Conv2d(1, 1, 1)
    layer(data["current_depth"].unsqueeze(0)).mean().backward()
    assert layer.weight.grad is not None
    assert not staged.exists()
    assert Path(row["gt"]).exists()
    # A changed source under the same canonical ID is a conflict, not a new cache hit.
    h5_file(tmp_path / "replacement.h5")
    manifest["samples"][-1]["current_rgb"] = str(tmp_path / "replacement.h5")
    write_json(config["paths"]["manifest"], manifest)
    with pytest.raises(ValueError, match="Settings changed"):
        plan.prepare_plan(config, base=base)


def test_migration_keeps_prediction_inode_and_resumes_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr("explore_a.cache.migration.assert_idle", lambda: None)
    old = root_path(tmp_path, "legacy")
    write_json(old / "recipe.json", {"id": "legacy"})
    write_json(old / "jobs/tum__room/job.json", {"recipe": "legacy"})
    payload = old / "frames/probe.npz"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"untouched prediction")
    inode, stamp = payload.stat().st_ino, payload.stat().st_mtime_ns
    write_json(old / ".migration.json", {"from": "legacy", "to": "glob"})
    target = root_path(tmp_path)
    old.rename(target)  # Simulate power loss immediately after the directory rename.
    migrate(tmp_path, "legacy", "glob")
    assert old.is_symlink() and old.resolve() == target
    assert read_json(target / "jobs/tum__room/job.json")["recipe"] == "glob"
    assert (target / "frames/probe.npz").stat().st_ino == inode
    assert (target / "frames/probe.npz").stat().st_mtime_ns == stamp
    assert migrate(tmp_path, "legacy", "glob")["already_migrated"]
    conflict = root_path(tmp_path, "other")
    conflict.mkdir()
    with pytest.raises(ValueError, match="Destination"):
        migrate(tmp_path, "other", "glob")


def test_legacy_numeric_normalization_rejects_model_and_canvas_changes(prepared):
    config, _, root, _ = prepared
    modern = read_json(root / "recipe.json")
    legacy = {**modern, "schema": 3, "data": config["data"]}
    assert_compatible(legacy, modern)
    import copy
    for key in ("data", "weights"):
        changed = copy.deepcopy(modern)
        if key == "data":
            changed[key]["canvas_indoor"] = [100, 100]
        else:
            changed[key]["raft"]["bytes"] += 1
        with pytest.raises(ValueError, match="numeric recipe"):
            assert_compatible(legacy, changed)
