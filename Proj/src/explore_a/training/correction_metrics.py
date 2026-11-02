"""Detached per-frame correction diagnostics, aggregated equally within each source."""
from collections import defaultdict

import torch


class CorrectionMetrics:
    def __init__(self, fraction):
        self.fraction = fraction
        self.values = defaultdict(list)
        self.names = ["global_mean", "local_std", "upper_fraction", "lower_fraction"]

    @torch.no_grad()
    def add(self, output, rows):
        if "global_correction" not in output:
            return
        for index, row in enumerate(rows):
            valid = output["visibility"][index] > 0
            if not valid.any():
                continue
            total = output["correction"][index][valid].float()
            local = output["local_correction"][index][valid].float()
            near = self.fraction * .95
            values = torch.stack([output["global_correction"][index].float().mean(),
                                  local.std(unbiased=False),
                                  (total >= near).float().mean() if near else total.mean() * 0,
                                  (total <= -near).float().mean() if near else total.mean() * 0])
            if "temporal_correction" in output:
                extra = ("spatial_correction", "temporal_correction",
                         "temporal_global_increment", "temporal_local_increment",
                         "temporal_global_gate", "temporal_local_gate")
                self.names = [*self.names[:4], *[name + "_mean" for name in extra]]
                selected = []
                for name in extra:
                    value = output[name][index].float()
                    selected.append(value[valid].mean() if value.shape == valid.shape
                                    else value.mean())
                values = torch.cat([values, torch.stack(selected)])
            self.values[row["dataset"]].append(values.cpu().tolist())

    def finish(self):
        return {source: {name: sum(row[i] for row in rows) / len(rows)
                         for i, name in enumerate(self.names)}
                for source, rows in self.values.items()}
