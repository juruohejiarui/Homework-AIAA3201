"""Host-aware status, graceful stop and explicit retries, independent of GPU models."""
import json
from pathlib import Path

from depth_project.io import read_json, write_json
from explore_a.cache import remote
from explore_a.cache.assignments import HOSTS, job_folder, local_state, owner
from explore_a.cache.locks import process_lock
from explore_a.cache.policy import ensure_eligible
from explore_a.cache.catalog import refresh_catalog


def selected_hosts(root, host=None):
    if host in HOSTS:
        return (host,)
    path = root / "scheduler.json"
    return tuple(read_json(path)["workers"]) if path.exists() else HOSTS


def status(root, hosts):
    states, details = {}, {}
    if "dgx" in hosts:
        states = json.loads(remote.control("status", root.name))
        details = json.loads(remote.control("details", root.name))
    local, assignments = {}, {}
    for folder in sorted((root / "jobs").glob("*")):
        if not folder.is_dir():
            continue
        host = owner(root, folder.name)
        if host is None and folder.name in states:
            host = "dgx"
        record = {"host": host, "resume_at": str(folder),
                  "state": local_state(root, folder.name) if host == "local"
                  else states.get(folder.name, "pending")}
        for filename in ("local.failed.json", "transfer.failed.json"):
            if (folder / filename).exists():
                record[filename] = read_json(folder / filename)
        if (folder / "local.progress.json").exists():
            record["progress"] = read_json(folder / "local.progress.json")
        if (folder / "local.complete.json").exists():
            report = read_json(folder / "local.complete.json")
            local[folder.name] = {"bytes": report["bytes"], "pairs": report["pairs"],
                                  "host": report.get("producer", {}).get("host", "dgx"),
                                  "generation_seconds": report["seconds"],
                                  "pairs_per_second": report["pairs"] /
                                  max(report["seconds"], 1e-9)}
            record["state"] = "complete"
        assignments[folder.name] = record
    catalog = refresh_catalog(root, states)
    return {"local_root": str(root), "workers": hosts, "remote": states,
            "cache_status_csv": str(root / "cache_status.csv"),
            "skipped_length": sum(not row["eligible"] for row in catalog),
            "remote_details": details, "assignments": assignments, "local": local,
            "local_bytes": sum(row["bytes"] for row in local.values()),
            "completed_pairs": sum(row["pairs"] for row in local.values())}


def stop(root, base, hosts, all_hosts=False):
    if all_hosts:
        write_json(root / "pipeline.stop.json", {"stop": True})
    for host in hosts:
        write_json(root / f"dispatch.{host}.stop.json", {"stop": True})
        if host == "local":
            write_json(Path(base).expanduser() / "stop.json", {"stop_after_current_job": True})
        else:
            remote.control("stop", root.name)
    return {"stop_after_current_job": hosts, "coordinator_stop": all_hosts}


def retry(root, base, name=None, hosts=HOSTS):
    with process_lock(Path(base).expanduser() / "coordinator.lock", "Cache coordinator"):
        return _retry(root, base, name, hosts)


def _retry(root, base, name, hosts):
    states = json.loads(remote.control("status", root.name)) if "dgx" in hosts else {}
    if name is None:
        names = {path.parent.name for filename in ("local.failed.json", "transfer.failed.json")
                 for path in (root / "jobs").glob(f"*/{filename}")}
        names.update(name for name, state in states.items() if state == "failed")
    else:
        from explore_a.cache.paths import component
        names = {component(name)}
    for name in sorted(names):
        ensure_eligible(root, name)
        folder = job_folder(root, name)
        if not (folder / "job.json").exists():
            raise ValueError(f"Unknown task: {name}")
        host = owner(root, name) or "dgx"  # Pre-upgrade jobs belong to the original producer.
        if host not in hosts:
            raise ValueError(f"Task {name} belongs to {host}; enable that host for retry")
        if host == "local":
            with process_lock(Path(base).expanduser() / "worker.lock", "Local GPU worker"):
                for filename in ("local.failed.json", "local.generating.json",
                                 "local.complete.json",
                                 "local.generated.json", "local.progress.json"):
                    (folder / filename).unlink(missing_ok=True)
        else:
            remote.control("retry", root.name, name)
            (folder / "local.complete.json").unlink(missing_ok=True)
            (folder / "dgx.pruned.json").unlink(missing_ok=True)
        (folder / "transfer.failed.json").unlink(missing_ok=True)
    return {"retried": sorted(names), "resume": "restart worker and pipeline; owner is retained"}
