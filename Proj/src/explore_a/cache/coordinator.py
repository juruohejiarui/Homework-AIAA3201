"""Single local coordinator dynamically dispatches whole sequences to either GPU host."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from depth_project.io import read_json, write_json
from explore_a.cache import remote
from explore_a.cache.assignments import HOSTS, adopt, job_folder, owner
from explore_a.cache.locks import process_lock
from explore_a.cache.plan import prepare_plan
from explore_a.cache.policy import eligible_names
from explore_a.cache.scheduling import completed, dispatch, done, failures
from explore_a.cache.storage import require_space
from explore_a.cache.transfers import upload_controls
from explore_a.observability.progress import message


def worker_hosts(value):
    hosts = value.split(",") if isinstance(value, str) else list(value)
    if not hosts or len(set(hosts)) != len(hosts) or any(host not in HOSTS for host in hosts):
        raise ValueError("--workers must be local, dgx, or local,dgx")
    return tuple(host for host in HOSTS if host in hosts)


def run(config, recipe, base, minimum_gib=50, workers=("dgx",)):
    base, workers = Path(base).expanduser(), worker_hosts(workers)
    with process_lock(base / "coordinator.lock", "Cache coordinator"):
        root = prepare_plan(config, recipe, base)
        require_space(root, minimum_gib)
        if "dgx" not in workers and read_json(root / "dispatch.origin.json")["legacy"]:
            if not (root / "dispatch.migrated.json").exists():
                raise ValueError("Legacy DGX assignments are unknown. First resume with "
                                 "--workers local,dgx to adopt remote jobs; no SSH was attempted.")
        if "dgx" in workers:
            upload_controls(root)
        (root / "pipeline.stop.json").unlink(missing_ok=True)
        for host in workers:
            (root / f"dispatch.{host}.stop.json").unlink(missing_ok=True)
        write_json(root / "scheduler.json", {"workers": workers, "started": time.time()})
        _coordinate(root, recipe, workers, minimum_gib)


def _coordinate(root, recipe, workers, minimum_gib):
    names = [row["job"] for row in read_json(root / "plan.json")["jobs"]]
    pending, last_report = {}, 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        while not (root / "pipeline.stop.json").exists():
            for name, (action, future) in list(pending.items()):
                if future.done():
                    try:
                        future.result()
                        (job_folder(root, name) / "transfer.failed.json").unlink(missing_ok=True)
                    except Exception as error:
                        write_json(job_folder(root, name) / "transfer.failed.json",
                                   {"action": action, "error": str(error), "time": time.time()})
                        raise
                    del pending[name]
            # Offline mode deliberately performs no SSH, including discovery and status calls.
            states = json.loads(remote.control("status", recipe)) if "dgx" in workers else {}
            adopt(root, names, states)
            active = eligible_names(root, names)
            if "dgx" in workers:
                write_json(root / "dispatch.migrated.json", {"remote_adopted": True})
            unavailable = [name for name in active if owner(root, name) not in (*workers, None)
                           and (not (job_folder(root, name) / "local.complete.json").exists() or
                                (owner(root, name) == "dgx" and
                                 not (job_folder(root, name) / "dgx.pruned.json").exists()))]
            if unavailable:
                raise ValueError("Unfinished tasks/cleanup belong to disabled workers: "
                                 f"{unavailable}")
            invalid = [name for name in active if states.get(name) == "pruned" and
                       not (job_folder(root, name) / "local.complete.json").exists()]
            if invalid:
                raise ValueError(f"Local receipts missing after remote cleanup; retry: {invalid}")
            failed = failures(root, names, states)
            if failed:
                raise ValueError("Cache jobs failed; no new jobs dispatched. "
                                 f"Inspect/retry: {failed}")
            if done(root, names, states):
                message("Cache pipeline complete", root)
                return
            dispatch(root, names, states, workers, pending, pool, minimum_gib)
            if time.monotonic() - last_report > 30:
                message("CACHE pipeline", {"complete": len(completed(root, names)),
                                           "total": len(names), "eligible": len(active),
                                           "skipped_length": len(names) - len(active),
                                           "workers": workers,
                                           "transfers": {name: item[0]
                                                         for name, item in pending.items()}})
                last_report = time.monotonic()
            time.sleep(2)
    message("Pipeline stopped; completed outputs retained. Restart workers/pipeline to resume.")
