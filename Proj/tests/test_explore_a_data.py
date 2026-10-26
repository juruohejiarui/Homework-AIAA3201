"""Leakage protection and actual consecutive RGB association on small synthetic folders."""
import json
from pathlib import Path

import pytest

from explore_a.data.history import evaluation_history
from explore_a.data.kitti import prepare_kitti
from explore_a.data.splits import validate_rows
from explore_a.download.tum_links import ArchiveLinks, category
from explore_a.training.sampling import StepBatches


def put(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_history_uses_paths_in_records_and_normalizes_drive_alias(tmp_path):
    drive = "2011_09_26_drive_0002"
    path = tmp_path / "run.json"
    put(path, json.dumps({"manifest": {"dataset": "kitti", "frames": [
        f"/another/export/{drive}/image_03/data/0000000004.png"]}}))
    result = evaluation_history([tmp_path])
    assert result == {drive + "_sync": [str(path)]}


def test_kitti_excludes_whole_drive_both_cameras_and_uses_raw_previous(tmp_path):
    names = [f"2011_09_26_drive_{i:04d}_sync" for i in range(1, 5)]
    for name in names:
        for camera in ("image_02", "image_03"):
            images = tmp_path / "raw/2011_09_26" / name / camera / "data"
            put(images / "0000000001.png")
            put(images / "0000000002.png")
            for split in ("train", "val"):
                put(tmp_path / "depth" / split / name / "proj_depth/groundtruth" / camera
                    / "0000000002.png")
    history = {names[0]: ["past/run.json"]}
    rows, _ = prepare_kitti(tmp_path, {"seed": 2026, "dev_fraction": .15}, history, {})
    assert {r["camera"] for r in rows if r["split"] == "test"} == {"image_02", "image_03"}
    assert all(r["sequence"] != names[0] for r in rows if r["split"] != "test")
    assert all(Path(r["previous_rgb"]).name == "0000000001.png" for r in rows)
    validate_rows(rows, history)
    aliases = {names[0]: "same-place", names[1]: "same-place"}
    separate, _ = prepare_kitti(tmp_path, {"seed": 2026, "dev_fraction": .15}, history, aliases)
    assert all(r["sequence"] != names[1] for r in separate if r["split"] != "test")
    validate_rows(separate, history)
    rows[0]["previous_rgb"] = rows[-1]["previous_rgb"]
    with pytest.raises(ValueError, match="canonical frame"):
        validate_rows(rows, history)


def test_sampler_resume_and_microbatch_preserve_examples():
    rows = [{"dataset": "kitti"}] * 10 + [{"dataset": "tum"}] * 2
    full = list(StepBatches(rows, 8, 8, 5, 0, 2026))
    resumed = list(StepBatches(rows, 8, 8, 5, 2, 2026))
    small = list(StepBatches(rows, 8, 4, 5, 2, 2026))
    assert full[2:] == resumed
    assert [small[i] + small[i + 1] for i in range(0, len(small), 2)] == resumed
    assert all(len({rows[i]["dataset"] for i in batch}) == 1 for batch in full)


def test_tum_link_classification_and_deduplication():
    url = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz"
    parser = ArchiveLinks()
    parser.feed(f'<a href="{url}">tgz</a><a href="{url}">duplicate</a>')
    assert len(parser.links) == 1
    assert category("rgbd_dataset_freiburg3_walking_xyz.tgz") == "ordinary"
    assert category("rgbd_dataset_freiburg3_walking_xyz_validation.tgz") == "validation"
    assert category("rgbd_dataset_freiburg3_checkerboard_large.tgz") == "calibration"
