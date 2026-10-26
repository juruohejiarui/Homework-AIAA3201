"""Resolve portable RGB IDs either to managed staging or unchanged local source files."""
from pathlib import Path

from explore_a.cache.storage import check_inventory, safe_relative


class RGBInputs:
    def __init__(self, job, staging=None, sources=None):
        self.staging = Path(staging) if staging is not None else None
        self.sources = sources
        if sources is None:
            check_inventory(self.staging, job["inputs"])
        else:
            if set(sources) != set(job["inputs"]):
                raise ValueError("Local RGB mapping differs from portable job inputs")
            for relative, size in job["inputs"].items():
                safe_relative(relative)
                record = sources[relative]
                path = Path(record["source"])
                if (not path.is_absolute() or not path.is_file() or record["bytes"] != size
                        or path.stat().st_size != size):
                    raise ValueError(f"Missing or changed local RGB: {path}")

    def __call__(self, relative):
        safe_relative(relative)
        if self.sources is not None:
            return Path(self.sources[relative]["source"])
        path = self.staging / relative
        if not path.resolve().is_relative_to(self.staging.resolve()):
            raise ValueError(f"Input escapes staging: {path}")
        return path
