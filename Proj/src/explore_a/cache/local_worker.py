"""Local GPU worker: resolve original RGBs directly and commit to the final cache root."""
from pathlib import Path
import time

from depth_project.io import read_json, write_json
from explore_a.cache.assignments import local_state, owner
from explore_a.cache.locks import process_lock
from explore_a.cache.plan import load_job
from explore_a.cache.policy import ensure_eligible, skipped
from explore_a.cache.storage import check_inventory
from explore_a.cache.worker_runtime import WorkerRuntime
from explore_a.observability.progress import message


def generate(root, name, producer, minimum_gib=50):
    ensure_eligible(root, name)
    if owner(root, name) != "local":
        raise ValueError(f"Task is not assigned to this local worker: {name}")
    folder = root / "jobs" / name
    job = load_job(root, name)
    sources = read_json(folder / "sources.json")
    report = producer.generate(root, job, minimum_gib=minimum_gib, sources=sources, host="local")
    if (report["job"] != name or report["recipe"] != job["recipe"] or
            set(report["sizes"]) != set(job["outputs"])):
        raise ValueError("Local producer output inventory differs from assigned task")
    check_inventory(root, report["sizes"])
    write_json(folder / "local.complete.json", report)
    (folder / "local.generating.json").unlink(missing_ok=True)
    (folder / "local.progress.json").unlink(missing_ok=True)


def run(base, once=False):
    base = Path(base).expanduser()
    with process_lock(base / "worker.lock", "Local GPU worker"):
        (base / "stop.json").unlink(missing_ok=True)
        models = WorkerRuntime()
        message("Local worker waiting for assigned sequence jobs", base)
        while not (base / "stop.json").exists():
            worked = False
            for ready in sorted((base / "explore_a").glob("*/jobs/*/local.ready.json")):
                root, name = ready.parents[2], ready.parent.name
                if skipped(root, name):
                    continue
                if local_state(root, name) not in ("ready", "generating"):
                    continue
                worked = True
                try:
                    write_json(ready.parent / "local.generating.json", {"job": name})
                    generate(root, name, models.get(root))
                except Exception as error:
                    write_json(ready.parent / "local.failed.json",
                               {"error": str(error), "time": time.time()})
                    write_json(base / "stop.json", {"failed_job": name})
                    message("LOCAL CACHE FAILED", name, error)
                if (base / "stop.json").exists():
                    break
            if once:
                break
            if not worked:
                time.sleep(2)
