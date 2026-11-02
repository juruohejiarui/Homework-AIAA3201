"""Stateless per-update sampling makes resume independent of raw-input prefetch."""
from collections import defaultdict
import random

import torch
from torch.utils.data import default_collate

TRAIN_KEYS = {"current_rgb", "previous_rgb", "current_depth", "previous_depth", "image_valid",
              "gt", "gt_valid", "previous_gt", "previous_gt_valid", "forward", "backward",
              "teacher_visibility", "previous_image_valid", "current_image_valid"}


def collate(samples):
    # Optional poses/K differ by source and never enter the network or training loss.
    keys = set(samples[0]) & TRAIN_KEYS
    return default_collate([{key: sample[key] for key in keys} for sample in samples])


def source_indices(rows):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["dataset"]].append(index)
    return dict(groups)


class StepBatches:
    def __init__(self, rows, effective_batch, microbatch, steps, start, seed, source_weights=None,
                 sampling_mode="fix"):
        # fix means fixed source probabilities, not a fixed subset or fixed batch.
        if sampling_mode not in ("fix", "video_sqrt"):
            raise ValueError("Unknown training sampling mode: " + str(sampling_mode))
        self.sampling_mode = sampling_mode
        self.groups = source_indices(rows)
        self.sources = sorted(self.groups)
        self.effective_batch, self.microbatch = effective_batch, microbatch
        self.steps, self.start, self.seed = steps, start, seed
        self.weights = None
        self.videos = {}
        if sampling_mode == "video_sqrt":
            for source, indices in self.groups.items():
                videos = defaultdict(list)
                for index in indices:
                    # Both KITTI cameras belong to one physical recording, not two videos.
                    videos[rows[index]["sequence"]].append(index)
                self.videos[source] = list(videos.values())
        if source_weights is not None:
            import math
            if set(source_weights) != set(self.sources) or any(
                    not math.isfinite(v) or v <= 0 for v in source_weights.values()):
                raise ValueError("Source weights must match nonempty training datasets")
            self.weights = [source_weights[s] for s in self.sources]

    def __iter__(self):
        for step in range(self.start, self.steps):
            rng = random.Random(self.seed + step)
            # Retain the original RNG path for configurations without explicit weights.
            source = (rng.choice(self.sources) if self.weights is None else
                      rng.choices(self.sources, weights=self.weights, k=1)[0])
            if self.sampling_mode == "fix":
                selected = rng.choices(self.groups[source], k=self.effective_batch)
            else:
                selected = self.video_batch(source, rng)
            for offset in range(0, len(selected), self.microbatch):
                yield selected[offset:offset + self.microbatch]

    def video_batch(self, source, rng):
        videos = self.videos[source]
        available, selected = [], []
        while len(selected) < self.effective_batch:
            if not available:
                # Small synthetic datasets may have fewer videos than the effective batch.
                available = list(range(len(videos)))
            weights = [len(videos[index]) ** .5 for index in available]
            video = rng.choices(available, weights=weights, k=1)[0]
            available.remove(video)
            selected.append(rng.choice(videos[video]))
        return selected

    def __len__(self):
        return (self.steps - self.start) * self.effective_batch // self.microbatch


def validation_batches(rows, batch):
    for indices in source_indices(rows).values():
        for offset in range(0, len(indices), batch):
            yield indices[offset:offset + batch]


def to_device(data, device):
    return {name: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for name, value in data.items()}
