"""CPU-only raw decoding and bounded prefetch; GPU models stay in the main process."""
from contextlib import closing

import numpy as np
import torch

from explore_a.data.reading import (
    depth_range, fit_depth, fit_image, read_frame, resize_intrinsics,
)
from explore_a.execution import Timings, prefetch
from explore_a.training.sampling import collate, validation_batches


def read_pair(row, settings):
    canvas = settings["canvas_kitti" if row["dataset"] == "kitti" else "canvas_indoor"]
    images, fitted, data = {}, {}, {}
    minimum, maximum = depth_range(row["dataset"], settings)
    for previous, prefix in ((False, "current"), (True, "previous")):
        rgb, gt = read_frame(row, previous)
        images[previous] = rgb
        fitted[previous], valid = fit_image(rgb, canvas)
        data[prefix + "_rgb"] = fitted[previous].transpose(2, 0, 1)
        data[prefix + "_image_valid"] = valid[None]
        if not previous:
            data["image_valid"] = valid[None]
            if row.get("K") is not None:
                data["K"] = resize_intrinsics(row["K"], rgb.shape, canvas).astype(np.float32)
        target = np.zeros(canvas, np.float32) if gt is None else fit_depth(gt, canvas)
        key = "previous_gt" if previous else "gt"
        data[key] = target[None]
        data[key + "_valid"] = (np.isfinite(target) & (target > minimum)
                                & (target < maximum))[None]
    if row.get("pose") is not None:
        data["pose"] = np.array(row["pose"], np.float32)
        data["previous_pose"] = np.array(row["previous_pose"], np.float32)
    return {"row": row, "images": images, "fitted": fitted, "canvas": canvas,
            "data": {key: torch.from_numpy(np.ascontiguousarray(value))
                     for key, value in data.items()}}


class OnlinePairs:
    def __init__(self, config, rows, runtime, flow, teacher=False, phase="evaluation"):
        self.config, self.rows, self.runtime = config, rows, runtime
        self.flow, self.teacher = flow, teacher
        self.phase = phase

    def __len__(self):
        return len(self.rows)

    def raw_stream(self, indices=None, flows=None, teacher_flows=(), native=False, submit=True):
        settings = self.config["online"]
        indices = range(len(self.rows)) if indices is None else indices
        scope = object()
        jobs = ((index, self.rows[index], self.runtime.prepare(
            self.rows[index], index, {f:f in teacher_flows for f in (flows or [self.flow])},
            phase=self.phase, native=native, scope=scope)) for index in indices)

        def load(job):
            index, row, ticket = job
            try:
                raw = read_pair(row, self.config["data"])
                raw["frozen_ticket"] = ticket
                if submit:
                    self.runtime.submit_decoded(raw)
                return index, raw
            except BaseException:
                self.runtime.cancel_ticket(ticket)
                raise

        def stream():
            try:
                with self.runtime.input_phase(self.phase):
                    yield from prefetch(jobs, load, settings["read_workers"], settings["prefetch"],
                                        Timings("raw input"))
            finally:
                self.runtime.cancel_prepared(scope)
        return stream()

    def iter_batches(self, batches=None, ordered=True):
        if batches is None:
            batches = validation_batches(self.rows, self.config["training"]["microbatch"])
        # Flatten without letting the reader consume the complete, potentially infinite schedule.
        scope = object()
        def jobs():
            occurrence = getattr(batches, "start", 0) * self.config["training"]["effective_batch"]
            for batch in batches:
                for offset, index in enumerate(batch):
                    ticket = self.runtime.prepare(self.rows[index], occurrence,
                                                  {self.flow:self.teacher},
                                                  phase=self.phase, scope=scope)
                    yield index, offset == len(batch) - 1, ticket
                    occurrence += 1

        settings = self.config["online"]
        def load(job):
            index, end, ticket = job
            try:
                raw = read_pair(self.rows[index], self.config["data"])
                raw["frozen_ticket"] = ticket
                self.runtime.submit_decoded(raw)
                return index, end, raw
            except BaseException:
                self.runtime.cancel_ticket(ticket)
                raise

        loaded = prefetch(jobs(), load, settings["read_workers"], settings["prefetch"],
                          Timings("raw batches"))
        try:
            with self.runtime.input_phase(self.phase), closing(loaded):
                indices, samples = [], []
                for index, end, raw in loaded:
                    samples.append(self.runtime.compute(raw, self.flow, self.teacher, ordered))
                    indices.append(index)
                    if end:
                        yield indices, collate(samples)
                        indices, samples = [], []
        finally:
            self.runtime.cancel_prepared(scope)
