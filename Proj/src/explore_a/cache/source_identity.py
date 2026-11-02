"""Bounded metadata checks prevent a shared canonical key from reading another RGB source."""
from functools import lru_cache
from pathlib import Path

from depth_project.io import read_json
from explore_a.cache.paths import job_id


class CacheSources:
    def __init__(self, root):
        self.root = Path(root)
        self.records = lru_cache(maxsize=32)(self._load)

    def _load(self, name):
        folder = self.root / "jobs" / name
        job = read_json(folder / "job.json")
        return {row["id"]: row for row in job["rows"]}, read_json(folder / "sources.json")

    def validate(self, row):
        """Return False for an unregistered job; conflicting registered identities are errors."""
        try:
            rows, sources = self.records(job_id(row))
        except FileNotFoundError:
            return False
        fixed = rows.get(row["id"])
        if fixed is None:
            raise ValueError(f"Unregistered pair in existing cache job: {row['id']}")
        for key in ("dataset", "sequence", "camera", "previous_id", "current_id"):
            if fixed[key] != row[key]:
                raise ValueError(f"Conflicting cached frame pair: {row['id']}")
        for prefix in ("previous", "current"):
            record = sources[fixed[prefix + "_rgb"]]
            source = Path(row[prefix + "_rgb"])
            if (source.resolve() != Path(record["source"]).resolve()
                    or source.stat().st_size != record["bytes"]):
                raise ValueError(f"Conflicting cached RGB source: {row['id']}")
        return True
