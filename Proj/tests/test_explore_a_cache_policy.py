"""RGB length boundaries, skipped dispatch and narrowly scoped remote cache deletion."""
from pathlib import Path
import shutil

import pytest

from depth_project.io import read_json, write_json
from explore_a.cache.assignments import assign
from explore_a.cache.catalog import refresh_catalog
from explore_a.cache.discard import discard_length
from explore_a.cache.policy import count_rgb, refresh_policy, skipped
from explore_a.cache.producer import Producer
from explore_a.cache.scheduling import dispatch, done, failures
from test_explore_a_distributed import prepared as prepared
from test_explore_a_pipeline import fake_backbones as fake_backbones


class Pool:
    def submit(self, *args):
        pytest.fail("No transfer may be scheduled for skipped jobs")


def test_rgb_counter_sums_both_cameras_and_ignores_other_files(tmp_path):
    drive = tmp_path / "drive"
    for camera, count in (("image_02", 2), ("image_03", 3)):
        folder = drive / camera / "data"
        folder.mkdir(parents=True)
        for index in range(count):
            (folder / f"{index}.png").touch()
        (folder / "timestamps.txt").touch()
    job = {"rows": [{"dataset": "kitti"}]}
    assert count_rgb(job, {"a": {"source": str(drive / "image_02/data/0.png")}}) == 5


@pytest.mark.parametrize("count,eligible", [(3999, True), (4000, True), (4001, False)])
def test_threshold_keeps_manifest_and_existing_outputs(prepared, monkeypatch, count, eligible):
    config, _, root, name = prepared
    plan_before = (root / "plan.json").read_bytes()
    manifest_before = Path(config["paths"]["manifest"]).read_bytes()
    monkeypatch.setattr("explore_a.cache.policy.count_rgb", lambda *args: count)
    policy = refresh_policy(root, config)
    assert policy["videos"][name]["eligible"] is eligible
    assert skipped(root, name) is not eligible
    assert (root / "plan.json").read_bytes() == plan_before
    assert Path(config["paths"]["manifest"]).read_bytes() == manifest_before
    if not eligible:
        assign(root, name, "local")
        write_json(root / "jobs" / name / "local.failed.json", {"error": "interrupted"})
        assert not failures(root, [name], {})
        assert done(root, [name], {})
        dispatch(root, [name], {}, ("local", "dgx"), {}, Pool(), 0)
        assert not (root / "jobs" / name / "local.ready.json").exists()


def test_remote_delete_preserves_other_video_and_local_cache(prepared, tmp_path):
    config, _, root, name = prepared
    names = [row["job"] for row in read_json(root / "plan.json")["jobs"]]
    remote = tmp_path / "remote" / root.name
    shutil.copytree(root, remote)
    producer = Producer(config, "cpu")
    for selected in names[:2]:
        folder = remote / "jobs" / selected
        job = read_json(folder / "job.json")
        producer.generate(remote, job, sources=read_json(folder / "sources.json"),
                          host="dgx", minimum_gib=0)
    target_job = read_json(remote / "jobs" / name / "job.json")
    relative = target_job["outputs"][0]
    local = root / relative
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(remote / relative, local)
    original = local.read_bytes()
    policy = read_json(remote / "cache_policy.json")
    policy["videos"][name].update(eligible=False, rgb_frames=4001, reason="skipped_length")
    write_json(remote / "cache_policy.json", policy)
    stage = remote / ".staging" / name
    stage.mkdir(parents=True)
    (stage / "managed.png").write_bytes(b"input copy")
    report = discard_length(remote, name)
    assert report["output_bytes_deleted"] > 0 and not stage.exists()
    assert all(not (remote / p).exists() for p in target_job["outputs"])
    assert local.read_bytes() == original
    other = read_json(remote / "jobs" / names[1] / "job.json")
    assert all((remote / p).exists() for p in other["outputs"])
    with pytest.raises(ValueError, match="eligible"):
        discard_length(remote, names[1])
    write_json(root / "cache_policy.json", policy)
    rows = refresh_catalog(root)
    assert next(row for row in rows if row["job"] == name)["local_cache"] == "partial"
