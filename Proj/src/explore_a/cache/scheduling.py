"""Bounded sequence dispatch; only the local coordinator may assign an unowned task."""
from depth_project.io import read_json
from explore_a.cache.assignments import (
    assign, available, job_folder, local_state, owner, paused, publish_local,
)
from explore_a.cache.storage import require_space
from explore_a.cache.policy import eligible_names
from explore_a.cache.transfers import acknowledge, pull, upload


def completed(root, names):
    return {name for name in names if (job_folder(root, name) / "local.complete.json").exists()}


def done(root, names, states):
    names = eligible_names(root, names)
    ready = completed(root, names)
    return all(name in ready and (owner(root, name) == "local" or states.get(name) == "pruned" or
                                 (job_folder(root, name) / "dgx.pruned.json").exists())
               for name in names)


def failures(root, names, states):
    names = eligible_names(root, names)
    return [name for name in names if local_state(root, name) == "failed" or
            (job_folder(root, name) / "transfer.failed.json").exists() or
            (owner(root, name) == "dgx" and states.get(name) == "failed")]


def dispatch(root, names, states, workers, pending, pool, minimum_gib):
    """Reserve before uploading/publishing. Running uploads remain sticky across a crash."""
    names = eligible_names(root, names)
    ready = completed(root, names)
    # Prefer pulling completed work over starting another input transfer.
    for name in names:
        if "dgx" not in workers or name in pending or len(pending) >= 2:
            continue
        if states.get(name) == "remote_complete":
            if name in ready:
                report = read_json(job_folder(root, name) / "local.complete.json")
                future = pool.submit(acknowledge, root, name, report)
            else:
                future = pool.submit(pull, root, name, minimum_gib)
            pending[name] = ("pull", future)
    if "local" in workers and not paused(root, "local"):
        occupied = any(owner(root, name) == "local" and name not in ready and
                       local_state(root, name) in ("ready", "generating") for name in names)
        if not occupied:
            candidates = available(root, names, "local", states)
            if candidates:
                require_space(root, minimum_gib)
                name = candidates[0]
                assign(root, name, "local")
                publish_local(root, name)
    if "dgx" not in workers or paused(root, "dgx"):
        return
    occupied = {name for name in names
                if states.get(name) in ("ready", "generating", "remote_complete")}
    occupied.update(name for name, (action, _) in pending.items() if action == "upload")
    for name in available(root, names, "dgx", states):
        if len(pending) >= 2 or len(occupied) >= 2:
            break
        if name in pending or states.get(name, "pending") != "pending":
            continue
        require_space(root, minimum_gib)
        assign(root, name, "dgx")
        pending[name] = ("upload", pool.submit(upload, root, name))
        occupied.add(name)
