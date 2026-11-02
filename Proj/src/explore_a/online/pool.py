"""Owned priority I/O workers: no CUDA, no deadline, no executor exit hook."""
from concurrent.futures import Future
from itertools import count
from queue import Empty, PriorityQueue
from threading import Thread


class Workers:
    def __init__(self, workers, name):
        self.queue, self.closed, self.sequence = PriorityQueue(), False, count()
        self.threads = [Thread(target=self._run, name=name, daemon=True) for _ in range(workers)]
        for thread in self.threads:
            thread.start()

    def _run(self):
        while True:
            _, _, future, function = self.queue.get()
            if future is None:
                return
            if not future.set_running_or_notify_cancel():
                continue
            try:
                result = function()
            except BaseException as error:
                future.set_exception(error)
            else:
                future.set_result(result)

    def submit(self, function, priority=0):
        if self.closed:
            raise RuntimeError('Frozen I/O pool closed')
        future = Future()
        self.queue.put((priority, next(self.sequence), future, function))
        return future

    def close(self):
        self.closed = True
        while True:
            try:
                _, _, future, _ = self.queue.get_nowait()
            except Empty:
                break
            if future is not None:
                future.cancel()
        for _ in self.threads:
            self.queue.put((float('inf'), next(self.sequence), None, None))
