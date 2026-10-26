"""Persistent single-coordinator task ownership, including adoption of pre-upgrade work."""
import time

from depth_project.io import read_json, write_json
from explore_a.cache.paths import component
from explore_a.config import bind_identity
from explore_a.cache.policy import skipped

HOSTS = ("local", "dgx")


def job_folder(root, name):
    return root / "jobs" / component(name)


def owner(root, name):
    path = job_folder(root, name) / "owner.json"
    if not path.exists():
        return None
    host = read_json(path)["host"]
    if host not in HOSTS:
        raise ValueError(f"Invalid task owner: {path}")
    return host


def assign(root, name, host):
    if host not in HOSTS:
        raise ValueError(host)
    # Called only while the coordinator lock is held. Ownership is never a timed lease.
    bind_identity(job_folder(root, name) / "owner.json", {"host": host})


def local_state(root, name):
    folder = job_folder(root, name)
    for marker, state in (("local.complete.json", "complete"), ("local.failed.json", "failed"),
                          ("local.generating.json", "generating"), ("local.ready.json", "ready")):
        if (folder / marker).exists():
            return state
    return "pending"


def adopt(root, names, remote_states):
    """Import legacy DGX work before assigning any new local tasks; never infer expiry."""
    for name in names:
        if skipped(root, name):
            continue
        folder = job_folder(root, name)
        host = owner(root, name)
        remote_active = name in remote_states
        if remote_states.get(name) == "pruned":
            write_json(folder / "dgx.pruned.json", {"job": name})
        local_owned = any((folder / f"local.{state}.json").exists()
                          for state in ("ready", "generating", "failed"))
        receipt = folder / "local.complete.json"
        if host == "local" and remote_active:
            raise ValueError(f"Conflicting DGX and local task ownership: {name}")
        if host is None:
            if remote_active:
                assign(root, name, "dgx")
            elif (folder / "transfer.failed.json").exists():
                assign(root, name, "dgx")
            elif local_owned:
                assign(root, name, "local")
            elif receipt.exists():
                report = read_json(receipt)
                assign(root, name, report.get("producer", {}).get("host", "dgx"))


def publish_local(root, name):
    if owner(root, name) != "local":
        raise ValueError(f"Local worker cannot claim another host's task: {name}")
    write_json(job_folder(root, name) / "local.ready.json", {"job": name, "time": time.time()})


def paused(root, host):
    return (root / f"dispatch.{host}.stop.json").exists()


def available(root, names, host, remote_states):
    """Keep sticky unfinished assignments first; then choose unowned sequence jobs."""
    def unfinished(name):
        return not (job_folder(root, name) / "local.complete.json").exists()
    return ([name for name in names if owner(root, name) == host and unfinished(name)] +
            [name for name in names if owner(root, name) is None and unfinished(name)
             and remote_states.get(name, "pending") == "pending"])
