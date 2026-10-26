"""Synthetic dual-host ownership, direct RGB generation, recovery and offline operation."""
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest

from depth_project.io import read_json, write_json
from explore_a.cache import management, scheduling
from explore_a.cache.assignments import adopt, assign, local_state, owner, publish_local
from explore_a.cache.coordinator import _coordinate
from explore_a.cache.local_worker import generate
from explore_a.cache.locks import process_lock
from explore_a.cache.paths import artifact
from explore_a.cache.plan import load_job
from explore_a.cache.producer import Producer
from explore_a.cache.storage import read_npz
from test_explore_a_distributed import prepared as prepared, fake_transport
from test_explore_a_pipeline import fake_backbones as fake_backbones


class ImmediatePool:
    def submit(self, fn, *args):
        result = Future()
        try:
            result.set_result(fn(*args))
        except Exception as error:
            result.set_exception(error)
        return result


def test_dynamic_dispatch_and_sticky_ownership(tmp_path, monkeypatch):
    names = [f"job{i}" for i in range(8)]
    states, uploads, pending = {}, [], {}
    def upload(root, name):
        uploads.append(name)
        states[name] = "ready"
    monkeypatch.setattr(scheduling, "upload", upload)
    pool = ImmediatePool()
    scheduling.dispatch(tmp_path, names, states, ("local", "dgx"), pending, pool, 0)
    assert owner(tmp_path, "job0") == "local"
    assert local_state(tmp_path, "job0") == "ready"
    assert uploads == ["job1", "job2"]
    assert all(owner(tmp_path, name) == "dgx" for name in uploads)
    pending.clear()
    # The simulated faster DGX finishes twice while the local job is still running.
    for name, next_name in (("job1", "job3"), ("job2", "job4")):
        write_json(tmp_path / "jobs" / name / "local.complete.json", {})
        states[name] = "pruned"
        scheduling.dispatch(tmp_path, names, states, ("local", "dgx"), pending, pool, 0)
        assert uploads[-1] == next_name
        assert owner(tmp_path, "job0") == "local"
        pending.clear()
    write_json(tmp_path / "jobs/job0/local.complete.json", {})
    scheduling.dispatch(tmp_path, names, states, ("local", "dgx"), pending, pool, 0)
    assert owner(tmp_path, "job5") == "local"
    # Ownership survives a crash before upload; a restarted local worker cannot take it.
    assign(tmp_path, "job6", "dgx")
    with pytest.raises(ValueError):
        assign(tmp_path, "job6", "local")
    with pytest.raises(ValueError):
        publish_local(tmp_path, "job6")
    assert owner(tmp_path, "job6") == "dgx"


def test_adopt_legacy_results_and_inflight_jobs(tmp_path):
    receipt = tmp_path / "jobs/old/local.complete.json"
    write_json(receipt, {"producer": {"machine": "aarch64"}})
    adopt(tmp_path, ["old", "done", "upload", "active"],
          {"done": "remote_complete", "upload": "pending", "active": "generating"})
    assert all(owner(tmp_path, name) == "dgx" for name in ("old", "done", "upload", "active"))
    assign(tmp_path, "ours", "local")
    with pytest.raises(ValueError, match="Conflicting"):
        adopt(tmp_path, ["ours"], {"ours": "generating"})


def test_direct_rgb_and_staged_outputs_match(prepared, tmp_path, monkeypatch):
    config, _, root, name = prepared
    job = load_job(root, name)
    assign(root, name, "local")
    publish_local(root, name)
    original_sources = read_json(root / "jobs" / name / "sources.json")
    original_sizes = {record["source"]: Path(record["source"]).stat().st_size
                      for record in original_sources.values()}
    generate(root, name, Producer(config, "cpu"), minimum_gib=0)
    report = read_json(root / "jobs" / name / "local.complete.json")
    assert report["producer"]["host"] == "local"
    assert not (root / ".upload").exists() and not (root / ".staging").exists()
    assert not (root / "jobs" / name / "remote.complete.json").exists()
    # Produce the same job through its portable paths using the fake remote transport.
    from explore_a.cache import transfers
    from explore_a.cache.settings import root_path
    remote_base = tmp_path / "dgx"
    fake_transport(monkeypatch, remote_base)
    transfers.upload_controls(root)
    transfers.upload(root, name)
    remote_root = root_path(remote_base)
    Producer(config, "cpu").generate(remote_root, job, remote_root / ".staging" / name, 0)
    for relative in job["outputs"]:
        local, remote = read_npz(root / relative), read_npz(remote_root / relative)
        for key in local:
            np.testing.assert_array_equal(local[key], remote[key])
            assert local[key].dtype == np.float32
    assert all(Path(path).stat().st_size == size for path, size in original_sizes.items())


def test_local_failure_no_receipt_then_resume(prepared):
    config, _, root, name = prepared
    assign(root, name, "local")
    publish_local(root, name)
    producer = Producer(config, "cpu")
    teacher = producer.runtime.teacher
    def interrupted(*args):
        raise RuntimeError("disk interruption")
    producer.runtime.teacher = interrupted
    with pytest.raises(RuntimeError, match="interruption"):
        generate(root, name, producer, 0)
    assert not (root / "jobs" / name / "local.complete.json").exists()
    job = load_job(root, name)
    path = artifact(root, "frames", job["rows"][0])
    stamp = path.stat().st_mtime_ns
    producer.runtime.teacher = teacher
    generate(root, name, producer, 0)
    assert (root / "jobs" / name / "local.complete.json").exists()
    assert path.stat().st_mtime_ns == stamp


def test_local_only_coordinator_and_management_never_ssh(prepared, monkeypatch):
    config, base, root, _ = prepared
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline local mode called SSH")
    monkeypatch.setattr(management.remote, "control", forbidden)
    names = [row["job"] for row in read_json(root / "plan.json")["jobs"]]
    producer = Producer(config, "cpu")
    def tick(_):
        for name in names:
            if owner(root, name) == "local" and local_state(root, name) == "ready":
                generate(root, name, producer, 0)
    monkeypatch.setattr("explore_a.cache.coordinator.time.sleep", tick)
    _coordinate(root, root.name, ("local",), 0)
    result = management.status(root, ("local",))
    assert result["completed_pairs"] == 6 and result["remote"] == {}
    assert all(value["host"] == "local" for value in result["local"].values())
    assert management.stop(root, base, ("local",), True)["coordinator_stop"]
    name = names[0]
    write_json(root / "jobs" / name / "local.failed.json", {"error": "retry"})
    management.retry(root, base, name, ("local",))
    assert local_state(root, name) == "ready" and owner(root, name) == "local"


def test_process_locks_reject_second_coordinator_or_gpu_worker(tmp_path):
    for name in ("coordinator.lock", "worker.lock"):
        with process_lock(tmp_path / name, name):
            with pytest.raises(ValueError, match="already active"):
                with process_lock(tmp_path / name, name):
                    pytest.fail("Second process acquired the same role")
        with process_lock(tmp_path / name, name):
            pass  # A stopped/crashed owner does not leave an expiring lease.


def test_local_worker_loop_and_background_failure(prepared, monkeypatch):
    config, base, root, name = prepared
    assign(root, name, "local")
    publish_local(root, name)
    from explore_a.cache import local_worker
    from explore_a.cache import producer as producer_module
    real_write = producer_module.write_npz
    def broken(path, **values):
        if "forward" in values:
            raise OSError("background compression failed")
        real_write(path, **values)
    monkeypatch.setattr(producer_module, "write_npz", broken)
    real_producer = Producer
    monkeypatch.setattr(producer_module, "Producer",
                        lambda settings: real_producer(settings, "cpu"))
    real_generate = local_worker.generate
    monkeypatch.setattr(local_worker, "generate",
                        lambda root, name, producer: real_generate(root, name, producer, 0))
    local_worker.run(base, once=True)
    assert local_state(root, name) == "failed"
    assert not (root / "jobs" / name / "local.complete.json").exists()
    failure = read_json(root / "jobs" / name / "local.failed.json")
    assert "background compression" in failure["error"]
    monkeypatch.setattr(producer_module, "write_npz", real_write)
    management.retry(root, base, name, ("local",))
    local_worker.run(base, once=True)
    assert local_state(root, name) == "complete"


def test_space_pause_happens_before_new_assignment(tmp_path, monkeypatch):
    def no_space(*args):
        raise OSError("Space pause")
    monkeypatch.setattr(scheduling, "require_space", no_space)
    with pytest.raises(OSError, match="Space pause"):
        scheduling.dispatch(tmp_path, ["job"], {}, ("local",), {}, ImmediatePool(), 50)
    assert owner(tmp_path, "job") is None
    assert local_state(tmp_path, "job") == "pending"


def test_legacy_offline_mode_refuses_guessing_remote_ownership(prepared, monkeypatch):
    config, base, root, _ = prepared
    from explore_a.cache import coordinator
    write_json(root / "dispatch.origin.json", {"legacy": True})
    monkeypatch.setattr(coordinator, "prepare_plan", lambda *args: root)
    def forbidden(*args):
        pytest.fail("Legacy offline refusal must happen before SSH")
    monkeypatch.setattr(coordinator.remote, "control", forbidden)
    with pytest.raises(ValueError, match="Legacy DGX assignments"):
        coordinator.run(config, root.name, base, 0, workers=("local",))
