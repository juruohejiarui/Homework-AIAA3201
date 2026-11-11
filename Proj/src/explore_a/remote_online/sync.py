"""One-time service code sync; never exports datasets, weights or predictions."""
import argparse
from pathlib import Path
import shlex
import subprocess
from threading import Lock

from explore_a.config import load_config
from explore_a.remote_online.policy import settings

_synchronized = set()
_sync_lock = Lock()


def ensure_synchronized(config):
    from depth_project.models.provenance import source_signature
    policy = settings(config)
    with _sync_lock:
        key = policy["host"], policy["project"], source_signature()
        if key not in _synchronized:
            synchronize(config)
            _synchronized.add(key)


def synchronize(config):
    policy = settings(config)
    destination = policy["project"] + "/Proj/src"
    subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", policy["host"],
                    shlex.join(["mkdir", "-p", destination])], check=True)
    source_root = Path(__file__).resolve().parents[2]
    for package in ("explore_a", "depth_project"):
        subprocess.run(["rsync", "-a", "--delete", "--exclude=__pycache__", "--exclude=*.pyc",
                        str(source_root / package),
                        policy["host"] + ":" + destination + "/"], check=True)
    print("DGX memory service synchronized; datasets and predictions were not transferred.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    synchronize(load_config(args.config))


if __name__ == "__main__":
    main()
