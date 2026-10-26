"""Upload deduplicated RGB links, pull committed outputs, then acknowledge remote cleanup."""
from pathlib import Path
import shutil

from depth_project.io import read_json, write_json
from explore_a.cache import remote
from explore_a.cache.plan import load_job
from explore_a.cache.policy import ensure_eligible
from explore_a.cache.settings import REMOTE_ROOT, root_path
from explore_a.cache.storage import check_inventory, require_space, safe_relative


def prepare_upload(stage, sources):
    """Recreate interrupted link farms; only managed symlinks may be replaced.

    Original RGB paths come from the fixed sources manifest. A stale link is transport
    state, not an input identity change. Regular files and symlink directories remain errors.
    """
    stage = Path(stage)
    if stage.is_symlink():
        raise ValueError(f"Upload staging must be a real directory: {stage}")
    stage.mkdir(parents=True, exist_ok=True)
    for relative, record in sources.items():
        path = stage / safe_relative(relative)
        source = Path(record["source"])
        if not source.is_absolute() or not source.is_file():
            raise ValueError(f"Missing original RGB: {source}")
        if source.stat().st_size != record["bytes"]:
            raise ValueError(f"Changed original RGB size: {source}")
        parent = path.parent
        while parent != stage:
            if parent.is_symlink():
                raise ValueError(f"Upload directory is a symlink: {parent}")
            parent = parent.parent
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            if str(path.readlink()) == str(source):
                continue
            path.unlink()  # The link only; its old target is never touched.
        elif path.exists():
            raise ValueError(f"Unknown file inside upload staging: {path}")
        path.symlink_to(source)


def remote_root(root):
    return root_path(REMOTE_ROOT, read_json(Path(root) / "recipe.json")["id"])


def upload_controls(root):
    root = Path(root)
    target = remote_root(root)
    remote.ssh(["mkdir", "-p", target / "jobs"])
    for filename in ("recipe.json", "producer.yaml", "cache_policy.json"):
        if not (root / filename).exists():
            continue
        remote.rsync(root / filename, f"{remote.HOST}:{target}/{filename}")


def upload(root, name):
    root = Path(root)
    ensure_eligible(root, name)
    target = remote_root(root)
    job = load_job(root, name)
    sources = read_json(root / "jobs" / name / "sources.json")
    stage = root / ".upload" / name
    prepare_upload(stage, sources)
    staging = target / ".staging" / name
    folder = target / "jobs" / name
    remote.ssh(["mkdir", "-p", staging, folder])
    remote.rsync(root / "jobs" / name / "job.json", f"{remote.HOST}:{folder}/job.json")
    remote.control("space", job["recipe"], name)
    remote.rsync(str(stage) + "/", f"{remote.HOST}:{staging}/", ("-L",))
    result = remote.control("ready", job["recipe"], name)
    shutil.rmtree(stage)  # Symlink farm only; original RGBs are never removed.
    return result


def acknowledge(root, name, report):
    root = Path(root)
    target = remote_root(root) / "jobs" / name
    ack = {"recipe": report["recipe"], "job": name, "sizes": report["sizes"]}
    path = root / "jobs" / name / "local.ack.json"
    write_json(path, ack)
    remote.rsync(path, f"{remote.HOST}:{target}/local.ack.json")
    result = remote.control("prune", report["recipe"], name)
    write_json(root / "jobs" / name / "dgx.pruned.json", {"job": name})
    return result


def pull(root, name, minimum_gib=50):
    root = Path(root)
    ensure_eligible(root, name)
    job, target = load_job(root, name), remote_root(root)
    folder = root / "jobs" / name
    candidate = folder / "pull-report.json"
    remote.rsync(f"{remote.HOST}:{target}/jobs/{name}/remote.complete.json", candidate)
    report = read_json(candidate)
    if (report["recipe"] != job["recipe"] or report["job"] != name or
            set(report["sizes"]) != set(job["outputs"])):
        raise ValueError("Wrong remote recipe/job/output inventory")
    existing = sum((root / path).stat().st_size for path in report["sizes"]
                   if (root / path).is_file())
    require_space(root, minimum_gib, max(0, report["bytes"] - existing))
    listing = folder / "pull-files.txt"
    listing.write_text("".join(path + "\n" for path in sorted(report["sizes"])))
    remote.rsync(f"{remote.HOST}:{target}/", str(root) + "/", ("--files-from", str(listing)))
    check_inventory(root, report["sizes"])
    write_json(folder / "local.complete.json", report)
    return acknowledge(root, name, report)
