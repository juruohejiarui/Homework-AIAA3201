"""SSH/rsync primitives; compressed artifacts are never recompressed by the transport."""
from pathlib import Path
import shlex
import subprocess

from explore_a.cache.settings import REMOTE_PROJECT, REMOTE_ROOT

HOST = "Dgx-e"


def ssh(arguments, capture=True):
    command = shlex.join([str(value) for value in arguments])
    return subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, command], check=True,
                          text=True, stdout=subprocess.PIPE if capture else None).stdout


def control(action, recipe, job=None):
    arguments = ["bash", f"{REMOTE_PROJECT}/Proj/dgx_cache.sh", "remote", action,
                 "--recipe", recipe, "--base", REMOTE_ROOT]
    if job:
        arguments += ["--job", job]
    return ssh(arguments)


def rsync(source, target, extra=()):
    subprocess.run(["rsync", "-a", "--partial-dir=.rsync-partial", "--info=progress2",
                    *extra, str(source), str(target)], check=True)


def sync_project(project):
    probe = (
        "from pathlib import Path; import fcntl; "
        f"p=Path({REMOTE_ROOT!r})/'worker.lock'; "
        "f=p.open('r') if p.exists() else None; "
        "fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB) if f else None")
    ssh(["python3", "-c", probe])
    ssh(["mkdir", "-p", REMOTE_PROJECT])
    excluded = (".git", "__pycache__", ".pytest_cache", ".venv", "*.pyc", "*.tmp",
                "Proj/outputs", "Proj/logs", "Proj/data", "Proj/storage", "*.whl")
    extra = [flag for name in excluded for flag in ("--exclude", name)]
    rsync(str(Path(project)) + "/", f"{HOST}:{REMOTE_PROJECT}/", extra)
