"""Whole logging-window throughput, rather than a single sampled optimization step."""
import math
from statistics import median
import time

from explore_a.online.statistics import snapshot


class InputWindow:
    def __init__(self, runtime):
        self.before = snapshot(runtime)
        self.started = time.monotonic()
        self.durations = []
        self.current, self.finished = self.before, self.started

    def capture(self, runtime, duration):
        self.current, self.finished = snapshot(runtime), time.monotonic()
        self.durations.append(duration)

    def report(self):
        counts = {key: value - self.before.get(key, 0) for key, value in self.current.items()}
        wall = max(self.finished - self.started, 1e-9)
        ordered = sorted(self.durations)
        return {"steps": len(ordered), "wall_s": wall,
                "update_s_median": median(ordered),
                "update_s_p95": ordered[max(0, math.ceil(.95 * len(ordered)) - 1)],
                "remote_pairs_per_s": counts.get("remote_pairs", 0) / wall,
                "remote_components_per_s": counts.get("remote_consumed", 0) / wall,
                "counts": counts}
