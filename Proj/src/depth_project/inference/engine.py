"""Shared model execution. Timing excludes model loading and disk output."""
import fcntl
import time
from contextlib import contextmanager

import numpy as np
import torch

from depth_project.paths import OUTPUTS


@contextmanager
def gpu_session():
    OUTPUTS.mkdir(exist_ok=True)
    with (OUTPUTS / ".gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def predict(adapter, frames, fps=1):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        if hasattr(adapter, "predict_video"):
            depths = adapter.predict_video(frames, fps)
        else:
            depths = np.stack([adapter.predict(frame) for frame in frames])
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    depths = np.asarray(depths, dtype=np.float32)
    if depths.shape != frames.shape[:3] or not np.isfinite(depths).all():
        raise ValueError("Prediction shape or finite-value check failed")
    profile = {
        "frames": len(frames),
        "seconds": seconds,
        "seconds_per_frame": seconds / len(frames),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "parameters": sum(parameter.numel() for parameter in adapter.model.parameters()),
        "timing": "Preprocess + inference + CPU transfer, no warmup; not paper latency",
    }
    return depths, profile
