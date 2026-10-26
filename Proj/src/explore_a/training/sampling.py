"""Stateless per-update sampling makes resume independent of raw-input prefetch."""
from collections import defaultdict
import random

import torch
from torch.utils.data import default_collate

TRAIN_KEYS = {"current_rgb", "previous_rgb", "current_depth", "previous_depth", "image_valid",
              "gt", "gt_valid", "previous_gt", "previous_gt_valid", "forward", "backward",
              "teacher_visibility"}


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
    def __init__(self, rows, effective_batch, microbatch, steps, start, seed):
        self.groups = source_indices(rows)
        self.sources = sorted(self.groups)
        self.effective_batch, self.microbatch = effective_batch, microbatch
        self.steps, self.start, self.seed = steps, start, seed

    def __iter__(self):
        for step in range(self.start, self.steps):
            rng = random.Random(self.seed + step)
            source = rng.choice(self.sources)  # Equal probability PER DATASET, not per frame.
            selected = rng.choices(self.groups[source], k=self.effective_batch)
            for offset in range(0, len(selected), self.microbatch):
                yield selected[offset:offset + self.microbatch]

    def __len__(self):
        return (self.steps - self.start) * self.effective_batch // self.microbatch


def validation_batches(rows, batch):
    for indices in source_indices(rows).values():
        for offset in range(0, len(indices), batch):
            yield indices[offset:offset + batch]


def to_device(data, device):
    return {name: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for name, value in data.items()}
