"""Per-artifact FP32 cache reads with lazy online fallback and no prediction writes."""
from collections import Counter
import time

import numpy as np
import torch

from explore_a.cache.paths import artifact, frame_key
from explore_a.cache.reader import cache_spec, configured_root
from explore_a.cache.storage import read_npz
from explore_a.data.reading import fit_image
from explore_a.observability.progress import message
from explore_a.online.frozen import synchronize
from explore_a.online.lazy import LazyFrozen


class HybridRuntime:
    def __init__(self, config, flows, teacher=False, device="cuda"):
        self.root, self.spec = configured_root(config), cache_spec(config)
        self.device, self.flows = device, flows
        self.frozen = LazyFrozen(config, device)
        self.counts, self.times, self.warned = Counter(), Counter(), set()
        self.reset()

    def reset(self):
        self.previous_key, self.previous_depth = None, None

    def statistics(self):
        return {**self.counts, **self.times, "model_load_s": self.frozen.load_seconds}

    def _cached(self, path, shapes, category):
        started = time.perf_counter()
        try:
            values = read_npz(path)
            if set(values) != set(shapes):
                raise ValueError(f"Unexpected NPZ fields: {list(values)}")
            for key, shape in shapes.items():
                if values[key].shape != shape:
                    raise ValueError(f"Unexpected {key} shape: {values[key].shape}")
        except FileNotFoundError:
            return None
        except Exception as error:
            self.counts[category + "_invalid"] += 1
            if path not in self.warned:
                message("HYBRID invalid cache; computing online:", path, error)
                self.warned.add(path)
            return None
        else:
            self.counts[category + "_hit"] += 1
            return values
        finally:
            self.times["cache_read_s"] += time.perf_counter() - started

    def _online(self, category, function, *args):
        synchronize(self.device)
        started = time.perf_counter()
        result = function(*args)
        synchronize(self.device)
        self.times["online_" + category + "_s"] += time.perf_counter() - started
        self.counts[category + "_online"] += 1
        return result

    def _tensor(self, value, shape):
        value = np.asarray(value)
        if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid hybrid frontend output: {value.dtype}/{value.shape}")
        # Inputs saved by trainable convolutions must be ordinary tensors even if a
        # frontend was first initialized from an inference-mode development stage.
        with torch.inference_mode(False):
            return torch.from_numpy(value).to(self.device)

    @torch.no_grad()
    def depth_pair(self, raw, ordered=False):
        if not ordered:
            self.reset()
        row, shape = raw["row"], (1, *raw["canvas"])
        data = {key: value.to(self.device) for key, value in raw["data"].items()}
        for previous, prefix in ((True, "previous"), (False, "current")):
            key = frame_key(row, previous)
            if previous and ordered and key == self.previous_key:
                value = self.previous_depth
                self.counts["depth_reuse"] += 1
            else:
                cached = self._cached(artifact(self.root, "frames", row, previous),
                                      {"depth": shape}, "depth")
                if cached is None:
                    predicted = self._online("depth", self.frozen.depth, raw["images"][previous])
                    fitted, _ = fit_image(predicted, raw["canvas"])
                    array = fitted[None]
                else:
                    array = cached["depth"]
                value = self._tensor(array, shape)
            data[prefix + "_depth"] = value
        if ordered:
            self.previous_key, self.previous_depth = frame_key(row), data["current_depth"]
        return data

    @torch.no_grad()
    def add_flow(self, raw, base, flow, teacher=False):
        row, shape = raw["row"], (2, *raw["canvas"])
        category = "flow_" + flow
        fields = self._cached(artifact(self.root, f"flow/{flow}", row),
                              {"forward": shape, "backward": shape}, category)
        previous, current = raw["fitted"][True], raw["fitted"][False]
        if fields is None:
            forward, backward = self._online(category, self.frozen.flow, flow, previous, current)
            fields = {"forward": np.asarray(forward, dtype=np.float32),
                      "backward": np.asarray(backward, dtype=np.float32)}
        data = dict(base)
        for key in ("forward", "backward"):
            data[key] = self._tensor(fields[key], shape)
        if teacher:
            category, shape = "teacher_" + flow, (1, *raw["canvas"])
            cached = self._cached(artifact(self.root, f"teacher/{flow}", row),
                                  {"visibility": shape}, category)
            value = (cached["visibility"] if cached is not None else
                     self._online(category, self.frozen.visibility, previous, current,
                                  fields["backward"]))
            data["teacher_visibility"] = self._tensor(np.asarray(value, dtype=np.float32), shape)
        return data

    def compute(self, raw, flow, teacher=False, ordered=False):
        return self.add_flow(raw, self.depth_pair(raw, ordered), flow, teacher)

    def timed_depth(self, raw):
        synchronize(self.device)
        started = time.perf_counter()
        result = self.depth_pair(raw, ordered=True)
        synchronize(self.device)
        return result, time.perf_counter() - started
