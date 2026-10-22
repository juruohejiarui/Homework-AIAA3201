"""Check the shared volume once at the boundary of a writing operation."""
import json
import subprocess
from pathlib import Path

VOLUME = Path("/media/hjr/新加卷")
VOLUME_UUID = "9E04A88104A85DD1"


def require_dataset_volume():
    result = subprocess.run(["findmnt", "--json", "--target", str(VOLUME),
                             "--output", "TARGET,UUID,OPTIONS"], capture_output=True, text=True, check=True)
    mount, = json.loads(result.stdout)["filesystems"]
    if Path(mount["target"]) != VOLUME or mount.get("uuid") != VOLUME_UUID:
        raise RuntimeError("The dataset volume is not mounted at the expected location")
    if "rw" not in mount["options"].split(","):
        raise RuntimeError("The dataset volume is read-only")
