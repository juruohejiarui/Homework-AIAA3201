"""Full Raw source discovery, official MD5 behavior and scene-level NYUv2 isolation."""
import hashlib
import io
from pathlib import Path
import zipfile

import h5py
import numpy as np
import pytest
from scipy.io import savemat

from depth_project.io import write_json
from explore_a.config import DEFAULT_CONFIG, load_config
from explore_a.data import nyu
from explore_a.data.splits import validate_rows
from explore_a.download import nyu as downloader
from explore_a.download import nyu_links
from explore_a.download.nyu_integrity import target_path
from explore_a.download.nyu_links import RAW_PREFIX, RawLinks, parse_md5


def test_official_zip_md5_pairing_and_wrong_filename_rejection():
    base = "https://horatio.cs.nyu.edu" + RAW_PREFIX
    parser = RawLinks()
    parser.feed(f'<a href="{base}basements.zip">zip</a>'
                f'<a href="{base}basements.md5">md5</a>'
                '<a href="toolbox_nyu_depth_v2.zip">toolbox</a>')
    entries = parser.entries()
    assert len(entries) == 1 and entries[0]["name"] == "basements.zip"
    digest = "58c333f15aa9cebf8a2352b3cb4f2d43"
    assert parse_md5(digest + "  basements.zip\n", "basements.zip") == digest
    with pytest.raises(ValueError, match="filename"):
        parse_md5(digest + "  another.zip", "basements.zip")


def test_full_discovery_keeps_official_classrooms_part_missing_from_page(tmp_path, monkeypatch):
    base = "https://horatio.cs.nyu.edu" + RAW_PREFIX
    page = f'<a href="{base}basements.zip">zip</a><a href="{base}basements.md5">md5</a>'
    prior = tmp_path / "previous.json"
    write_json(prior, {"archives": [{"name": name, "url": base + name}
                                    for name in ("basements.zip", "classrooms.zip")]})
    monkeypatch.setattr(nyu_links, "KNOWN_ARCHIVES", prior)
    monkeypatch.setattr(nyu_links, "get_text", lambda url: page)
    monkeypatch.setattr(nyu_links, "resolve_entry", lambda entry: entry)
    entries = {entry["name"]: entry for entry in nyu_links.discover(workers=1)}
    assert set(entries) == {"basements.zip", "classrooms.zip"}
    assert entries["classrooms.zip"]["md5_url"] == base + "classrooms.md5"
    assert entries["classrooms.zip"]["discovery"] \
        == "previous_official_manifest_not_on_current_page"


def raw_plan(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("office_0100/r-1.000-1.ppm", b"synthetic raw RGB")
    payload = buffer.getvalue()
    entry = {"name": "offices.zip", "url": "https://official.example/offices.zip",
             "bytes": len(payload), "md5": hashlib.md5(payload).hexdigest()}
    plan = {"output": str(tmp_path / "data"), "files": [entry]}
    path = tmp_path / "plan.json"
    write_json(path, plan)
    return path, plan, entry, payload


def test_download_verifies_with_aria2_and_explicit_verify_rehashes_once(tmp_path, monkeypatch):
    path, plan, entry, payload = raw_plan(tmp_path)
    calls = []
    def fake_aria(args, check):
        assert check and "--check-integrity=true" in args
        request = next(arg.split("=", 1)[1] for arg in args if arg.startswith("--input-file="))
        assert "checksum=md5=" + entry["md5"] in Path(request).read_text()
        target_path(plan, entry).write_bytes(payload)
        calls.append("aria2")
    monkeypatch.setattr(downloader.subprocess, "run", fake_aria)
    original = hashlib.file_digest
    hashes = []
    def tracked_digest(stream, algorithm):
        hashes.append(algorithm)
        return original(stream, algorithm)
    monkeypatch.setattr(hashlib, "file_digest", tracked_digest)
    downloader.download(path, jobs=2)
    assert calls == ["aria2"] and hashes == []
    downloader.verify(path)
    assert hashes == ["md5"]
    downloader.extract(path)
    assert hashes == ["md5"]  # Extraction reuses an unchanged official-verification receipt.
    extracted = Path(plan["output"]) / "raw/scenes/office_0100/r-1.000-1.ppm"
    assert extracted.read_bytes() == b"synthetic raw RGB"
    downloader.download(path, jobs=2)
    assert calls == ["aria2"] and hashes == ["md5"]
    corrupted = bytearray(payload)
    corrupted[0] ^= 1
    target_path(plan, entry).write_bytes(corrupted)
    with pytest.raises(ValueError, match="MD5 mismatch"):
        downloader.verify(path)


def test_existing_zip_is_checked_against_md5_not_only_size(tmp_path, monkeypatch):
    path, plan, entry, payload = raw_plan(tmp_path)
    target = target_path(plan, entry)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x" * len(payload))
    def forbidden_download(*args, **kwargs):
        raise AssertionError("Existing corrupt ZIP must fail official MD5 before any download")
    monkeypatch.setattr(downloader.subprocess, "run", forbidden_download)
    with pytest.raises(ValueError, match="MD5 mismatch"):
        downloader.download(path, jobs=1)


def test_all_raw_scenes_are_considered_and_test_room_variants_never_train(tmp_path, monkeypatch):
    scenes = ("bedroom_0001a", "bedroom_0001b", "office_0100", "kitchen_0101",
              "living_room_0102", "library_0104")
    with h5py.File(tmp_path / "nyu_depth_v2_labeled.mat", "w") as labeled:
        name = labeled.create_dataset("scene_name", data=np.array(
            [ord(c) for c in "bedroom_0001a"], dtype=np.uint16))
        references = labeled.create_dataset("scenes", (1, 1), dtype=h5py.ref_dtype)
        references[0, 0] = name.ref
    savemat(tmp_path / "splits.mat", {"testNdxs": np.array([[1]])})
    for name in scenes:
        folder = tmp_path / "raw" / name
        folder.mkdir(parents=True)
        for index, time in enumerate((1.000, 1.033, 1.066, 1.200)):
            (folder / f"r-{time:.3f}-{index}.ppm").touch()
            (folder / f"d-{time + .001:.3f}-{index}.pgm").touch()
    monkeypatch.setattr(nyu, "historical_scenes", lambda dataset: {"library_0104"})
    settings = load_config()["data"]
    rows, report = nyu.prepare_nyu(tmp_path, settings, {})
    assert len(rows) == len(scenes) * 2  # Every valid anchor, rather than labeled images only.
    assert {r["sequence"] for r in rows} == set(scenes)
    protected = {"bedroom_0001a", "bedroom_0001b", "library_0104"}
    assert all(r["split"] == "test" for r in rows if r["sequence"] in protected)
    assert all(r["sequence"] not in protected for r in rows if r["split"] == "train")
    assert report["source"] == "official_raw" and report["rgb_gap"] == len(scenes)
    validate_rows(rows, {})
    aliased, _ = nyu.prepare_nyu(tmp_path, settings, {"office_0100": "bedroom_0001"})
    assert all(r["split"] == "test" for r in aliased if r["sequence"] == "office_0100")
    validate_rows(aliased, {})


def test_default_has_three_sources_and_scannet_stays_optional():
    assert set(load_config()["data"]["datasets"]) == {"nyuv2", "tum", "kitti"}
    four = load_config(DEFAULT_CONFIG.with_name("mixed_scannet.yaml"))
    assert set(four["data"]["datasets"]) == {"nyuv2", "tum", "kitti", "scannet"}
