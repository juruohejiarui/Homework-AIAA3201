"""Synthetic-only distributed producer/consumer, transport receipts and safe cleanup."""
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pytest
import torch

from depth_project.io import read_json, write_json
from explore_a.cache import plan, transfers, worker
from explore_a.cache.paths import artifact, expected
from explore_a.cache.producer import Producer
from explore_a.cache.reader import CachedRuntime
from explore_a.cache.settings import root_path
from explore_a.cache.storage import read_npz, write_npz
from explore_a.online.loading import read_pair
from test_explore_a_pipeline import online_experiment, fake_backbones as fake_backbones


@pytest.fixture
def prepared(tmp_path):
    config = online_experiment(tmp_path)
    config["data"]["datasets"] = ["tum"]
    manifest = read_json(config["paths"]["manifest"])
    manifest["settings"] = config["data"]
    from collections import Counter
    manifest["counts"] = dict(Counter(f"{row['dataset']}/{row['split']}"
                                      for row in manifest["samples"]))
    write_json(config["paths"]["manifest"], manifest)
    base = tmp_path / "local"
    root = plan.prepare_plan(config, base=base)
    name = read_json(root / "plan.json")["jobs"][0]["job"]
    return config, base, root, name


def fake_transport(monkeypatch, remote_base):
    monkeypatch.setattr(transfers, "REMOTE_ROOT", str(remote_base))
    def ssh(args, **_):
        assert args[:2] == ["mkdir", "-p"]
        for value in args[2:]:
            Path(value).mkdir(parents=True, exist_ok=True)
        return ""

    def rsync(source, target, extra=()):
        source = Path(str(source).removeprefix("Dgx-e:"))
        target = Path(str(target).removeprefix("Dgx-e:"))
        target.parent.mkdir(parents=True, exist_ok=True)
        if "--files-from" in extra:
            listing = Path(extra[extra.index("--files-from") + 1])
            for line in listing.read_text().splitlines():
                dst = target / line
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / line, dst)
        elif source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)

    def control(action, recipe, job=None):
        if action == "space":
            return "{}"  # Small test filesystem bypasses only the production 50GiB reserve.
        return json.dumps(worker.control(action, remote_base, recipe, job))
    monkeypatch.setattr(transfers.remote, "ssh", ssh)
    monkeypatch.setattr(transfers.remote, "rsync", rsync)
    monkeypatch.setattr(transfers.remote, "control", control)


def test_fp32_maximum_compression_and_portable_grouping(prepared):
    config, base, root, name = prepared
    job = plan.load_job(root, name)
    assert len(job["inputs"]) == 3
    assert not any("depth" in name for name in job["inputs"])
    assert all(not Path(row["current_rgb"]).is_absolute() for row in job["rows"])
    assert job["history_drives"]
    assert job["outputs"] == expected(job["rows"])
    path = root / "probe.npz"
    value = np.arange(32, dtype=np.float32).reshape(1, 4, 8) / 7
    write_npz(path, depth=value)
    np.testing.assert_array_equal(read_npz(path)["depth"], value)
    with zipfile.ZipFile(path) as archive:
        assert archive.getinfo("depth.npy").compress_type == zipfile.ZIP_DEFLATED
    assert read_json(root / "recipe.json")["storage"] == "fp32-npz-deflate-9"


def test_transfer_generate_pull_then_prune_and_read_locally(prepared, tmp_path, monkeypatch):
    config, base, root, name = prepared
    remote_base = tmp_path / "dgx"
    fake_transport(monkeypatch, remote_base)
    transfers.upload_controls(root)
    transfers.upload(root, name)
    remote_root = root_path(remote_base)
    assert worker.state(remote_root, name) == "ready"
    job = plan.load_job(remote_root, name)
    producer = Producer(config, device="cpu")
    report = producer.generate(remote_root, job, remote_root / ".staging" / name, minimum_gib=0)
    assert len(report["sizes"]) == 11 and report["producer"]["machine"]
    transfers.pull(root, name, minimum_gib=0)
    assert worker.state(remote_root, name) == "pruned"
    assert not (remote_root / ".staging" / name).exists()
    assert not any((remote_root / item).exists() for item in report["sizes"])
    assert all((root / item).exists() for item in report["sizes"])
    config["execution"] = {"backend": "cache"}
    config["cache"] = {"root": str(base), "recipe": read_json(root / "recipe.json")["id"]}
    runtime = CachedRuntime(config, ["raft"], teacher=True, device="cpu")
    original = read_json(config["paths"]["manifest"])["samples"]
    row = next(row for row in original if row["id"] == job["rows"][0]["id"])
    result = runtime.compute(read_pair(row, config["data"]), "raft", True)
    assert result["current_depth"].dtype == torch.float32
    np.testing.assert_array_equal(result["current_depth"].numpy(), 2)
    np.testing.assert_allclose(result["teacher_visibility"].numpy(), .8)
    assert Path(row["current_rgb"]).exists() and Path(row["gt"]).exists()
    # Deserialized frontends are ordinary tensors: heads can backpropagate without
    # loading a local frozen backbone or retaining an inference-mode tensor.
    from explore_a.training.prediction import make_fusion, predict
    model = make_fusion("unet", config)
    batch = {key: value.unsqueeze(0) for key, value in result.items()}
    prediction, _ = predict(model, batch, config, "rule", None)
    prediction["depth"].mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    artifact(root, "frames", row).unlink()
    with pytest.raises(FileNotFoundError):
        runtime.compute(read_pair(row, config["data"]), "raft", True)


def test_partial_pull_never_commits_or_prunes(prepared, tmp_path, monkeypatch):
    config, _, root, name = prepared
    remote_base = tmp_path / "dgx"
    fake_transport(monkeypatch, remote_base)
    transfers.upload_controls(root)
    transfers.upload(root, name)
    target = root_path(remote_base)
    job = plan.load_job(target, name)
    report = Producer(config, "cpu").generate(target, job, target / ".staging" / name, 0)
    (target / next(iter(report["sizes"]))).unlink()
    with pytest.raises(FileNotFoundError):
        transfers.pull(root, name, 0)
    assert not (root / "jobs" / name / "local.complete.json").exists()
    assert (target / ".staging" / name).exists()
    assert not (target / "jobs" / name / "pruned.json").exists()


def test_missing_ack_and_forged_inventory_cannot_delete_inputs(prepared, tmp_path, monkeypatch):
    config, _, root, name = prepared
    remote_base = tmp_path / "dgx"
    fake_transport(monkeypatch, remote_base)
    transfers.upload_controls(root)
    transfers.upload(root, name)
    target = root_path(remote_base)
    job = plan.load_job(target, name)
    Producer(config, "cpu").generate(target, job, target / ".staging" / name, 0)
    with pytest.raises(FileNotFoundError):
        worker.control("prune", remote_base, job["recipe"], name)
    write_json(target / "jobs" / name / "remote.complete.json", {"sizes": {"recipe.json": 1}})
    with pytest.raises(ValueError, match="canonical"):
        worker.control("prune", remote_base, job["recipe"], name)
    assert (target / "recipe.json").exists() and (target / ".staging" / name).exists()


def test_timm_attention_adapter_preserves_float32_formula():
    from timm.layers import Attention
    from explore_a.models.attention import configure_depth_attention
    model = Attention(32, num_heads=4).eval()
    inputs = torch.randn(2, 16, 32)
    with torch.no_grad():
        expected_value = model(inputs)
        policy = configure_depth_attention(model)
        result = model(inputs)
    torch.testing.assert_close(result, expected_value)
    assert policy.report()["calls"]["native_sdpa_fallback"] == 1


def test_worker_failed_job_resumes_without_reupload(prepared, tmp_path, monkeypatch):
    config, _, root, name = prepared
    remote_base = tmp_path / "dgx"
    fake_transport(monkeypatch, remote_base)
    transfers.upload_controls(root)
    transfers.upload(root, name)
    target = root_path(remote_base)
    real_producer = Producer
    fail = [True]

    def factory(settings):
        producer = real_producer(settings, "cpu")
        teacher = producer.runtime.teacher

        def predict(*args):
            if fail[0]:
                fail[0] = False
                raise RuntimeError("interrupted teacher")
            return teacher(*args)
        producer.runtime.teacher = predict
        return producer

    monkeypatch.setattr("explore_a.cache.producer.Producer", factory)
    worker.run(remote_base, once=True)
    assert worker.state(target, name) == "failed"
    assert list((target / "frames").rglob("*.npz"))
    worker.control("retry", remote_base, read_json(root / "recipe.json")["id"], name)
    worker.run(remote_base, once=True)
    assert worker.state(target, name) == "remote_complete"
    assert not (target / "jobs" / name / "failed.json").exists()


def test_kitti_paths_keep_date_drive_cameras_and_splits():
    from explore_a.cache.paths import relative, job_id
    sequence = "2011_09_26_drive_0001_sync"
    a = {"dataset": "kitti", "sequence": sequence, "camera": "image_02",
         "current_id": "0000000001", "previous_id": "0000000000"}
    b = {**a, "camera": "image_03"}
    assert job_id(a) == job_id(b)
    assert str(artifact(Path("/cache"), "frames", a)) == (
        f"/cache/frames/kitti/2011_09_26/{sequence}/image_02/data/0000000001.npz")
    assert artifact(Path("/cache"), "frames", a) != artifact(Path("/cache"), "frames", b)
    assert str(relative("flow/raft", f"kitti:{sequence}:image_03:0000000001")).endswith(
        "image_03/data/0000000001.npz")
