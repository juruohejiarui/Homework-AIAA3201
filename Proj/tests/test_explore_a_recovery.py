"""Power-loss recovery checks, using synthetic outputs and managed upload links only."""
import pytest

from depth_project.io import read_json
from explore_a.cache.audit import inspect_job
from explore_a.cache.producer import Producer
from explore_a.cache.transfers import prepare_upload
from test_explore_a_distributed import prepared as prepared
from test_explore_a_pipeline import fake_backbones as fake_backbones


def test_interrupted_upload_repairs_only_stale_links(tmp_path):
    source, old = tmp_path / "source.png", tmp_path / "old.png"
    source.write_bytes(b"rgb")
    old.write_bytes(b"do not remove")
    stage = tmp_path / "upload"
    stage.mkdir()
    link = stage / "frame.png"
    link.symlink_to(old)
    sources = {"frame.png": {"source": str(source), "bytes": 3}}
    prepare_upload(stage, sources)
    assert link.readlink() == source and old.read_bytes() == b"do not remove"
    prepare_upload(stage, sources)  # An identical retry is harmless.
    link.unlink()
    link.write_bytes(b"unknown local file")
    with pytest.raises(ValueError, match="Unknown file"):
        prepare_upload(stage, sources)
    assert link.read_bytes() == b"unknown local file"


def test_upload_refuses_directory_links_and_changed_raw(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"rgb")
    stage = tmp_path / "upload"
    stage.mkdir()
    (stage / "camera").symlink_to(tmp_path, target_is_directory=True)
    sources = {"camera/frame.png": {"source": str(source), "bytes": 3}}
    with pytest.raises(ValueError, match="directory is a symlink"):
        prepare_upload(stage, sources)
    (stage / "camera").unlink()
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed original"):
        prepare_upload(stage, sources)


def test_audit_distinguishes_missing_outputs_and_corrupt_npz(prepared):
    config, _, root, name = prepared
    folder = root / "jobs" / name
    job = read_json(folder / "job.json")
    sources = read_json(folder / "sources.json")
    report = Producer(config, "cpu").generate(root, job, sources=sources,
                                              host="local", minimum_gib=0)
    from depth_project.io import write_json
    write_json(folder / "local.complete.json", report)
    healthy = inspect_job(root, name, deep=True)
    assert healthy["files"] == len(job["outputs"]) and not healthy["bad"]
    path = root / job["outputs"][0]
    size = path.stat().st_size
    path.write_bytes(b"\x00" * size)
    assert not inspect_job(root, name)["bad"]  # Equal length does not prove ZIP integrity.
    broken = inspect_job(root, name, deep=True)
    assert len(broken["bad"]) == 1 and broken["bad"][0]["path"] == job["outputs"][0]
    path.unlink()
    assert inspect_job(root, name)["missing"] == [job["outputs"][0]]
    write_json(folder / "local.complete.json", {**report, "sizes": None})
    invalid_receipt = inspect_job(root, name)
    assert invalid_receipt["metadata_errors"] and not invalid_receipt["bad"]
