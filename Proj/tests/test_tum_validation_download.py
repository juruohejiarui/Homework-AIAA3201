"""Official validation selection and dispatch; never downloads archive payloads."""
import argparse
import pytest

from depth_project.io import write_json
from explore_a.download import tum, tum_validation


def entries():
    return [{"name": "rgbd_dataset_freiburg1_xyz" + suffix + ".tgz", "category": kind,
             "url": "https://cvg.cit.tum.de/rgbd/dataset/test" + suffix + ".tgz"}
            for suffix, kind in (("", "ordinary"), ("_validation", "validation"),
                                 ("_calibration", "calibration"))]


def test_selection_preserves_default_and_excludes_other_categories():
    rows = entries()
    assert tum.select_archives(rows, "ordinary") == rows[:1]
    assert tum.select_archives(rows, "validation") == rows[1:2]
    assert tum.select_archives(rows, "all") == rows
    with pytest.raises(ValueError, match="outside selected"):
        tum.select_archives(rows, "validation", [rows[0]["name"]])
    with pytest.raises(argparse.ArgumentTypeError):
        tum_validation.archive_name("../../freiburg1_xyz_validation")


def test_plan_and_download_dispatch(tmp_path, monkeypatch):
    rows = entries()
    monkeypatch.setattr(tum, "crawl", lambda: rows)
    monkeypatch.setattr(tum, "get_size", lambda e: {**e, "bytes": 123})
    path = tmp_path / "plan.json"
    tum.plan(path, tmp_path / "dataset", subset="validation")
    planned = tum_validation.validation_plan(path)
    assert [e["category"] for e in planned["files"]] == ["validation"]
    captured = []
    monkeypatch.setattr(tum.subprocess, "run", lambda cmd, **kw: captured.append(cmd))
    monkeypatch.setattr(tum, "complete", lambda p, e: bool(captured))
    tum.download(path, jobs=2)
    assert "--continue=true" in captured[0]
    request = path.with_suffix(".aria2.txt").read_text()
    assert "_validation.tgz" in request
    assert "_calibration.tgz" not in request
    with pytest.raises(ValueError, match="Plan exists"):
        tum.plan(path, tmp_path / "dataset", subset="validation")
    planned["files"] = rows[:1]
    write_json(path, planned)
    with pytest.raises(ValueError, match="non-validation"):
        tum_validation.validation_plan(path)
