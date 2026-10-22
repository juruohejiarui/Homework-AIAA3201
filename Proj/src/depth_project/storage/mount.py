"""Check the shared volume once at the boundary of a writing operation."""
import json
import subprocess
from depth_project.paths import DATASETS

VOLUME_UUID = "9E04A88104A85DD1"


def require_dataset_volume():
    result = subprocess.run(["findmnt", "--json", "--target", str(DATASETS.resolve()),
                             "--output", "TARGET,UUID,OPTIONS"], capture_output=True, text=True, check=True)
    mount, = json.loads(result.stdout)["filesystems"]
    if mount.get("uuid") != VOLUME_UUID:
        raise RuntimeError("The dataset volume is not mounted at the expected location")
    if "rw" not in mount["options"].split(","):
        raise RuntimeError("The dataset volume is read-only")
