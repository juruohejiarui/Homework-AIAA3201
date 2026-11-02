"""Bounded representative reads of existing cache files; never constructs frozen models."""
from collections import defaultdict
from pathlib import Path
import random
import zipfile

import torch

from depth_project.io import read_json
from explore_a.cache.paths import artifact
from explore_a.cache.reader import cache_spec, configured_root
from explore_a.cache.source_identity import CacheSources
from explore_a.frontend import field, shared_root
from explore_a.cache.storage import read_npz
from explore_a.online.loading import read_pair


def load_probe(config):
    root = configured_root(config)
    from explore_a.cache.reader import check_component
    check_component(cache_spec(config), "flow_farneback")
    sources, flows = CacheSources(root), CacheSources(shared_root(root))
    grouped = defaultdict(lambda: defaultdict(list))
    for row in read_json(config["paths"]["manifest"])["samples"]:
        if row["split"] == "train":
            grouped[row["dataset"]][row["sequence"]].append(row)
    selected, skipped, inventory = {}, [], {}
    for source in ("kitti", "tum", "nyuv2"):
        sequences = sorted(grouped[source])
        rng = random.Random(config["training"]["seed"])
        rng.shuffle(sequences)
        pairs = []
        for sequence in sequences[:12]:
            rows = sorted(grouped[source][sequence], key=lambda row: row["id"])
            row = rows[rng.randrange(len(rows))]
            paths = [artifact(root, "frames", row, True), artifact(root, "frames", row),
                     artifact(root, "flow/farneback", row)]
            if not all(path.is_file() for path in paths):
                skipped.append({"id": row["id"], "reason": "missing_cache"})
                continue
            # Identity conflicts are errors, not cache misses silently accepted by a diagnostic.
            if not sources.validate(row) or not flows.validate(row):
                skipped.append({"id": row["id"], "reason": "unregistered_cache"})
                continue
            try:
                previous, current, flow = [read_npz(path) for path in paths]
                raw = read_pair(row, config["data"])
                data = raw["data"]
                data.update(previous_depth=torch.from_numpy(previous[field(config)]),
                            current_depth=torch.from_numpy(current[field(config)]))
                data.update({key: torch.from_numpy(flow[key]) for key in ("forward", "backward")})
                h, w = raw["canvas"]
                for key in ("previous_depth", "current_depth", "forward", "backward"):
                    channels = 2 if key in ("forward", "backward") else 1
                    if tuple(data[key].shape) != (channels, h, w):
                        raise ValueError("Cached shape differs from canvas")
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as error:
                skipped.append({"id": row["id"], "reason": str(error)})
                continue
            for path in paths:
                stat = path.stat()
                inventory[str(path)] = [stat.st_size, stat.st_mtime_ns]
            pairs.append((row, data))
            if len(pairs) == 6:
                break
        if not pairs:
            raise ValueError(f"No usable existing cache for {source}; skipped={skipped}")
        selected[source] = pairs
    return selected, skipped, inventory


def unchanged(inventory):
    return all([Path(path).stat().st_size, Path(path).stat().st_mtime_ns] == before
               for path, before in inventory.items())
