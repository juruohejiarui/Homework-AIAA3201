"""Bounded result I/O: one ordered JSONL writer and a small CPU-only image writer pool."""
from contextlib import ExitStack
import json

from explore_a.execution import BackgroundWriter, Timings
from explore_a.evaluation.visualization import save_panel


def safe_id(value):
    if not value or "/" in value or "\\" in value or ".." in value:
        raise ValueError(f"Unsafe sample ID: {value}")
    return value.replace(":", "__")


class ResultSink:
    def __init__(self, settings):
        timing = Timings("evaluation result I/O")
        self.records = BackgroundWriter({**settings, "write_workers": 1}, timing)
        self.images = BackgroundWriter(settings, timing)
        self.streams, self.stack = {}, ExitStack()

    def __enter__(self):
        self.stack.enter_context(self.records)
        self.stack.enter_context(self.images)
        return self

    def open(self, root):
        root.mkdir(parents=True, exist_ok=True)
        # A failed rerun must not leave a stale success summary next to partial frame records.
        (root / "metrics.json").unlink(missing_ok=True)
        self.streams[root] = (root / "frames.jsonl").open("w")

    def record(self, root, record):
        def write():
            self.streams[root].write(json.dumps(record) + "\n")
        self.records.submit(write)

    def panel(self, path, data, output, maximum):
        # Only the few tensors used by this panel cross to CPU; no CUDA work in writer threads.
        fields = ("current_rgb", "image_valid", "gt", "gt_valid", "current_depth")
        images = {key: data[key].detach().to("cpu", copy=True) for key in fields}
        shown = {key: output[key].detach().to("cpu", copy=True)
                 for key in ("depth", "historical_weight", "visibility")}
        self.images.submit(save_panel, path, images, shown, maximum)

    def __exit__(self, kind, value, traceback):
        try:
            return self.stack.__exit__(kind, value, traceback)
        finally:
            for stream in self.streams.values():
                stream.close()
