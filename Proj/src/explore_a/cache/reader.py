"""Local cache consumer; raw RGB/GT stays local and no frozen model is constructed."""
import time

import torch

from depth_project.io import read_json
from explore_a.cache.paths import artifact, job_id
from explore_a.cache.settings import LOCAL_ROOT, RECIPE, root_path
from explore_a.cache.storage import read_npz


def configured_root(config):
    settings = config.get("cache", {})
    return root_path(settings.get("root", LOCAL_ROOT), settings.get("recipe", RECIPE))


def cache_spec(config):
    spec = read_json(configured_root(config) / "recipe.json")
    if spec["schema"] != 3 or spec["storage"] != "fp32-npz-deflate-9":
        raise ValueError("Unsupported cache format or precision")
    if spec["data"] != config["data"] or spec["geometry"] != config["geometry"]:
        raise ValueError("Cache preprocessing differs from the fixed experiment")
    return spec


class CachedRuntime:
    def __init__(self, config, flows, teacher=False, device="cuda"):
        self.root, self.device = configured_root(config), device
        self.spec, self.flows, self.teacher = cache_spec(config), flows, teacher
        self.checked = set()
        self.reset()

    def reset(self):
        pass

    def require_rows(self, rows):
        for job in {job_id(row) for row in rows}:
            report = read_json(self.root / "jobs" / job / "local.complete.json")
            if report["recipe"] != self.spec["id"] or report["job"] != job:
                raise ValueError(f"Mismatched local completion receipt: {job}")
            self.checked.add(job)

    def _array(self, path, key, shape):
        values = read_npz(path)
        value = values[key]
        if value.shape != shape:
            raise ValueError(f"Invalid cached shape at {path}: {value.shape} != {shape}")
        return torch.from_numpy(value).to(self.device)

    def depth_pair(self, raw, ordered=False):
        row = raw["row"]
        if job_id(row) not in self.checked:
            self.require_rows([row])
        data = {key: value.to(self.device) for key, value in raw["data"].items()}
        shape = (1, *raw["canvas"])
        for previous, prefix in ((True, "previous"), (False, "current")):
            data[prefix + "_depth"] = self._array(
                artifact(self.root, "frames", row, previous), "depth", shape)
        return data

    def add_flow(self, raw, base, flow, teacher=False):
        row, shape = raw["row"], (2, *raw["canvas"])
        values = read_npz(artifact(self.root, f"flow/{flow}", row))
        data = dict(base)
        for key in ("forward", "backward"):
            if values[key].shape != shape:
                raise ValueError(f"Invalid cached flow shape for {row['id']}")
            data[key] = torch.from_numpy(values[key]).to(self.device)
        if teacher:
            data["teacher_visibility"] = self._array(
                artifact(self.root, f"teacher/{flow}", row), "visibility", (1, *raw["canvas"]))
        return data

    def compute(self, raw, flow, teacher=False, ordered=False):
        return self.add_flow(raw, self.depth_pair(raw), flow, teacher)

    def timed_depth(self, raw):
        if self.device == "cuda":
            torch.cuda.synchronize()
        began = time.perf_counter()
        result = self.depth_pair(raw)
        if self.device == "cuda":
            torch.cuda.synchronize()
        return result, time.perf_counter() - began
