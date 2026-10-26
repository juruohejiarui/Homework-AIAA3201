"""Bounded CPU queues overlap loading and result writes with main-thread GPU inference."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Lock
import time


class Timings:
    def __init__(self, label):
        self.label, self.started = label, time.perf_counter()
        self.values, self.lock = {}, Lock()

    @contextmanager
    def measure(self, key):
        began = time.perf_counter()
        try:
            yield
        finally:
            with self.lock:
                self.values[key] = self.values.get(key, 0) + time.perf_counter() - began

    def report(self, count):
        wall = time.perf_counter() - self.started
        with self.lock:
            values = {key: round(value, 3) for key, value in self.values.items()}
        print(f"{self.label}: {count} new tasks, {count / max(wall, 1e-9):.2f} tasks/s, "
              f"wall={wall:.1f}s; overlapping worker/wait seconds={values}", flush=True)


def prefetch(items, load, workers, limit, timings):
    """Ordered loading with at most limit outstanding payloads, including completed results."""
    def timed(item):
        with timings.measure("read"):
            return load(item)

    source, pending = iter(items), deque()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            for _ in range(limit):
                item = next(source, None)
                if item is None:
                    break
                pending.append(pool.submit(timed, item))
            while pending:
                with timings.measure("read_wait"):
                    result = pending.popleft().result()
                yield result
                item = next(source, None)
                if item is not None:
                    pending.append(pool.submit(timed, item))
        finally:
            for future in pending:
                future.cancel()


class BackgroundWriter:
    """Bounded postprocessing/writes; context exit drains and propagates every worker failure."""
    def __init__(self, settings, timings):
        self.pool = ThreadPoolExecutor(max_workers=settings["write_workers"])
        self.limit, self.timings = settings["write_queue"], timings
        self.pending = deque()

    def __enter__(self):
        return self

    def submit(self, function, *args, **kwargs):
        if len(self.pending) >= self.limit:
            with self.timings.measure("write_wait"):
                self.pending.popleft().result()

        def work():
            with self.timings.measure("postprocess_write"):
                function(*args, **kwargs)
        self.pending.append(self.pool.submit(work))

    def __exit__(self, kind, value, traceback):
        try:
            for future in self.pending:
                future.result()
        finally:
            self.pool.shutdown(wait=True, cancel_futures=True)
