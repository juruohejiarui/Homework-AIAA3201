"""Shared model execution. Timing excludes model loading and disk output."""
import errno
import os
import time
from contextlib import contextmanager

import numpy as np
import torch

from depth_project.paths import OUTPUTS


@contextmanager
def gpu_session():
    OUTPUTS.mkdir(exist_ok=True)
    with (OUTPUTS / ".gpu.lock").open("a+b") as lock:
        if os.name != "nt":
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX)
            yield
            return

        import msvcrt

        lock.seek(0)
        while True:
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                time.sleep(0.25)
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def predict(adapter, frames, fps=1):
    from depth_project.models.api import ModelInput
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        parameters = {"fps": fps} if adapter.capability == "sequence" else {}
        depths = adapter.forward(ModelInput(rgb=frames), **parameters).numpy()
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
