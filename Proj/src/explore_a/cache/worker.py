"""Single-GPU producer queue; control actions are metadata-only and cannot delete raw data."""
from pathlib import Path
import shutil
import time

from depth_project.io import read_json, write_json
from explore_a.cache.paths import component
from explore_a.cache.plan import load_job
from explore_a.cache.policy import ensure_eligible, skipped
from explore_a.cache.settings import REMOTE_ROOT, root_path
from explore_a.cache.storage import check_inventory, safe_relative, require_space
from explore_a.cache.locks import process_lock
from explore_a.cache.worker_runtime import WorkerRuntime
from explore_a.observability.progress import message


def worker_lock(base):
    return process_lock(Path(base).expanduser() / "worker.lock", "DGX GPU worker")


def state(root, name):
    folder = root / "jobs" / component(name)
    if skipped(root, name):
        return "skipped_length"
    if (folder / "pruned.json").exists():
        return "pruned"
    if (folder / "remote.complete.json").exists():
        return "remote_complete"
    if (folder / "failed.json").exists():
        return "failed"
    if (folder / "generating.json").exists():
        return "generating"
    if (folder / "input.ready.json").exists():
        return "ready"
    return "pending"


def control(action, base, recipe, name=None):
    root = root_path(base, recipe)
    if action in ("status", "details"):
        jobs = root / "jobs"
        if not jobs.exists():
            return {}
        result = {}
        for folder in sorted(jobs.iterdir()):
            if not folder.is_dir():
                continue
            result[folder.name] = state(root, folder.name)
            if action == "details":
                result[folder.name] = {"state": result[folder.name],
                                       "resume_at": str(folder)}
                if (folder / "failed.json").exists():
                    result[folder.name]["failure"] = read_json(folder / "failed.json")
                if (folder / "dgx.progress.json").exists():
                    result[folder.name]["progress"] = read_json(folder / "dgx.progress.json")
                if (folder / "remote.complete.json").exists():
                    report = read_json(folder / "remote.complete.json")
                    result[folder.name].update(bytes=report["bytes"], pairs=report["pairs"],
                                               generation_seconds=report["seconds"],
                                               pairs_per_second=report["pairs"] /
                                               max(report["seconds"], 1e-9))
                if (folder / "discarded_length.json").exists():
                    result[folder.name]["historical_generated_bytes"] = result[folder.name].get(
                        "bytes", 0)
                    result[folder.name]["bytes"] = 0
                    result[folder.name]["discarded"] = read_json(folder / "discarded_length.json")
        return result
    if action == "stop":
        write_json(Path(base).expanduser() / "stop.json", {"stop_after_current_job": True})
        return {"stop_after_current_job": True}
    folder = root / "jobs" / component(name)
    if action == "space":
        job = load_job(root, name)
        require_space(root, 50, sum(job["inputs"].values()))
        return {"space_ready": True}
    if action == "ready":
        job = load_job(root, name)
        check_inventory(root / ".staging" / name, job["inputs"])
        if read_json(root / "recipe.json")["id"] != job["recipe"]:
            raise ValueError("Recipe mismatch")
        write_json(folder / "input.ready.json", {"recipe": recipe, "job": name})
        return {"state": "ready"}
    if action == "retry":
        ensure_eligible(root, name)
        if state(root, name) in ("generating", "failed"):
            with worker_lock(base):
                (folder / "generating.json").unlink(missing_ok=True)
        (folder / "failed.json").unlink(missing_ok=True)
        if (folder / "pruned.json").exists():
            for filename in ("pruned.json", "remote.complete.json", "input.ready.json",
                             "local.ack.json", "generating.json"):
                (folder / filename).unlink(missing_ok=True)
        return {"state": state(root, name)}
    if action == "discard-length":
        from explore_a.cache.discard import discard_length
        with worker_lock(base):
            return discard_length(root, name)
    if action == "prune":
        report = read_json(folder / "remote.complete.json")
        from explore_a.cache.paths import expected
        job = load_job(root, name)
        if set(report["sizes"]) != set(expected(job["rows"])):
            raise ValueError("Refusing cleanup: output inventory differs from canonical job IDs")
        ack = read_json(folder / "local.ack.json")
        if ack != {"recipe": recipe, "job": name, "sizes": report["sizes"]}:
            raise ValueError("No matching local acknowledgement; remote output is retained")
        for relative in report["sizes"]:
            path = root / safe_relative(relative)
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Output cleanup would escape the managed cache")
        for relative in report["sizes"]:
            (root / relative).unlink(missing_ok=True)
        staging = root / ".staging" / name
        if staging.is_symlink():
            raise ValueError("Refusing symlink staging cleanup")
        if staging.exists():
            shutil.rmtree(staging)
        write_json(folder / "pruned.json", {"recipe": recipe, "job": name})
        return {"state": "pruned"}
    raise ValueError(action)


def run(base=REMOTE_ROOT, once=False):
    base = Path(base).expanduser()
    with worker_lock(base):
        (base / "stop.json").unlink(missing_ok=True)
        models = WorkerRuntime()
        message("DGX worker waiting for uploaded sequence jobs", base)
        while not (base / "stop.json").exists():
            available = sorted((base / "explore_a").glob("*/jobs/*/input.ready.json"))
            worked = False
            for ready in available:
                root, name = ready.parents[2], ready.parent.name
                if state(root, name) not in ("ready", "generating"):
                    continue
                worked = True
                job = load_job(root, name)
                try:
                    producer = models.get(root)
                    write_json(ready.parent / "generating.json", {"job": name})
                    producer.generate(root, job, root / ".staging" / name)
                    (ready.parent / "generating.json").unlink(missing_ok=True)
                except Exception as error:
                    write_json(ready.parent / "failed.json",
                               {"error": str(error), "time": time.time()})
                    write_json(base / "stop.json", {"failed_job": name})
                    message("CACHE FAILED", name, error)
                if (base / "stop.json").exists():
                    break
            if once:
                break
            if not worked:
                time.sleep(2)
