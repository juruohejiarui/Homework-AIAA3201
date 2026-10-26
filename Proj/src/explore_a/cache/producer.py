"""One GPU worker generates only missing FP32 model outputs, then atomically commits a job."""
from contextlib import closing
import importlib.metadata as metadata
from pathlib import Path
import platform
import time

import cv2
import numpy as np
import torch

from depth_project.io import write_json
from explore_a.cache.inputs import RGBInputs
from explore_a.cache.policy import ensure_eligible
from explore_a.cache.paths import artifact, frame_key
from explore_a.cache.storage import inventory, require_space, write_npz
from explore_a.data.reading import fit_image
from explore_a.execution import BackgroundWriter, Timings, prefetch
from explore_a.models.attention import configure_depth_attention
from explore_a.observability.progress import Progress
from explore_a.online.frozen import FrozenRuntime


def decoded(row, inputs, config):
    images, fitted = {}, {}
    canvas = config["data"]["canvas_kitti" if row["dataset"] == "kitti" else "canvas_indoor"]
    for previous, prefix in ((False, "current"), (True, "previous")):
        path = inputs(row[prefix + "_rgb"])
        image = cv2.imread(str(path))
        if image is None:
            raise OSError(path)
        images[previous] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fitted[previous], _ = fit_image(images[previous], canvas)
    return {"row": row, "images": images, "fitted": fitted, "canvas": canvas, "data": {}}


class Producer:
    def __init__(self, config, device="cuda"):
        self.config, self.device = config, device
        self.runtime = FrozenRuntime(config, ["raft", "farneback"], True, device)
        self.attention = configure_depth_attention(self.runtime.depth.model)

    def generate(self, root, job, staging=None, minimum_gib=50, *, sources=None, host="dgx"):
        root = Path(root)
        ensure_eligible(root, job["job"])
        inputs = RGBInputs(job, staging, sources)
        require_space(root, minimum_gib)
        started = time.monotonic()
        last_progress = 0.
        processed = 0
        prior_attention = dict(self.attention.counts)
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        timing = Timings(f"{host} producer")
        done = set()
        options = self.config["online"]
        settings = {"write_workers": 4, "write_queue": 8}

        def load(row):
            return decoded(row, inputs, self.config)

        loaded = prefetch(job["rows"], load, options["read_workers"], options["prefetch"], timing)
        self.runtime.reset()
        with (closing(loaded), BackgroundWriter(settings, timing) as writer,
              Progress(self.config, f"CACHE {job['job']}", len(job["rows"])) as progress):
            for raw in loaded:
                row = raw["row"]
                require_space(root, minimum_gib)
                for previous in (True, False):
                    key = frame_key(row, previous)
                    path = artifact(root, "frames", row, previous)
                    if key not in done and not path.exists():
                        with torch.inference_mode():
                            depth = self.runtime.depth.predict(raw["images"][previous])
                        if not np.isfinite(depth).all() or (depth <= 0).any():
                            raise ValueError(f"Invalid Depth Pro output: {key}")
                        fitted, _ = fit_image(depth.astype(np.float32), raw["canvas"])
                        writer.submit(write_npz, path, depth=fitted[None])
                    done.add(key)
                    if hasattr(self.runtime.depth, "focal_predictions"):
                        self.runtime.depth.focal_predictions.clear()
                previous, current = raw["fitted"][True], raw["fitted"][False]
                for flow in ("farneback", "raft"):
                    field_path = artifact(root, f"flow/{flow}", row)
                    teacher_path = artifact(root, f"teacher/{flow}", row)
                    fields = None
                    if not field_path.exists():
                        fields = self.runtime.flows[flow](previous, current)
                        writer.submit(write_npz, field_path, forward=fields[0].astype(np.float32),
                                      backward=fields[1].astype(np.float32))
                    if not teacher_path.exists():
                        if fields is None:
                            from explore_a.cache.storage import read_npz
                            backward = read_npz(field_path)["backward"]
                        else:
                            backward = fields[1]
                        output = self.runtime.teacher(previous, current, backward)
                        writer.submit(write_npz, teacher_path,
                                      visibility=output["visibility"].astype(np.float32))
                progress.update(1, sequence=row["sequence"], camera=row["camera"])
                processed += 1
                now = time.monotonic()
                if now - last_progress >= 30:
                    write_json(root / "jobs" / job["job"] / f"{host}.progress.json", {
                        "host": host, "processed_pairs": processed,
                        "total_pairs": len(job["rows"]),
                        "seconds": now - started, "pairs_per_second": processed / (now - started),
                        "updated": time.time()})
                    last_progress = now
        sizes = inventory(root, job["outputs"])
        attention = self.attention.report()
        attention["calls"] = {key: count - prior_attention.get(key, 0)
                              for key, count in self.attention.counts.items()}
        report = {"recipe": job["recipe"], "job": job["job"], "sizes": sizes,
                  "bytes": sum(sizes.values()), "pairs": len(job["rows"]),
                  "seconds": time.monotonic() - started, "attention": attention,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30
                  if self.device == "cuda" else None,
                  "producer": {"host": host, "machine": platform.machine(),
                               "python": platform.python_version(),
                               "torch": torch.__version__,
                               "packages": {name: metadata.version(name) for name in
                                            ("numpy", "timm", "opencv-python")},
                               "torchvision": metadata.version("torchvision"),
                               "device": torch.cuda.get_device_name() if self.device == "cuda"
                               else "cpu"}}
        filename = "remote.complete.json" if host == "dgx" else "local.generated.json"
        write_json(root / "jobs" / job["job"] / filename, report)
        (root / "jobs" / job["job"] / f"{host}.progress.json").unlink(missing_ok=True)
        return report
