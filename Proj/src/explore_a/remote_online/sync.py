"""One-time service code sync; never exports datasets, weights or predictions."""
import argparse
from pathlib import Path
import shlex
import subprocess

from explore_a.config import load_config
from explore_a.remote_online.policy import settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    policy = settings(load_config(args.config))
    destination = policy["project"] + "/Proj/src"
    subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", policy["host"],
                    shlex.join(["mkdir", "-p", destination])], check=True)
    source_root = Path(__file__).resolve().parents[2]
    for package in ("explore_a", "depth_project"):
        subprocess.run(["rsync", "-a", "--delete", "--exclude=__pycache__", "--exclude=*.pyc",
                        str(source_root / package),
                        policy["host"] + ":" + destination + "/"], check=True)
    print("DGX memory service synchronized; datasets and predictions were not transferred.")


if __name__ == "__main__":
    main()
