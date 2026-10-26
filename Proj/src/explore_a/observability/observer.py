"""Training events -> append-only logs, rollback-aware views, TensorBoard and small plots."""
from contextlib import contextmanager
import fcntl
from pathlib import Path

from depth_project.io import read_json
from explore_a.evaluation.state import evaluation_state
from explore_a.observability import settings
from explore_a.observability.history import FILES, append_record, load_history, new_session
from explore_a.observability.plots import plot_history
from explore_a.observability.scalars import replay, write_scalars
from explore_a.observability.table import export_csv


@contextmanager
def run_lock(root):
    path = Path(root) / "observability/writer.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(
                f"Active observer owns {root}; do not rebuild TensorBoard concurrently")
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def tensorboard_writer(root, flush_s):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise ValueError("TensorBoard missing. In aiaa3201 run: python -m pip install tensorboard; "
                         "or set observability.tensorboard: false") from error
    # A session restart removes stale scalar views, then replays only valid historical points.
    return SummaryWriter(log_dir=str(Path(root) / "tensorboard"), purge_step=0,
                         flush_secs=flush_s)


class RunObserver:
    def __init__(self, root, config, resume_step):
        self.root, self.options, self.resume_step = Path(root), settings(config), resume_step
        self.writer, self.history, self.lock = None, None, None
        self.baseline, self.last_plot_step = None, None
        self.last_dev_sources = {}
        self.last_dev_input_stats = {}

    def __enter__(self):
        self.lock = run_lock(self.root)
        self.lock.__enter__()
        try:
            with evaluation_state():
                new_session(self.root, self.resume_step)
                self.history = load_history(self.root)
                path = self.root / "heldout_baseline.json"
                self.baseline = read_json(path) if path.exists() else None
                if self.options["tensorboard"]:
                    self.writer = tensorboard_writer(self.root, self.options["tb_flush_s"])
                    replay(self.writer, self.history, self.baseline)
        except BaseException:
            if self.writer is not None:
                self.writer.close()
            self.lock.__exit__(None, None, None)
            raise
        return self

    def _put(self, kind, record, persist=True):
        if persist:
            append_record(self.root / FILES[kind], record)
        self.history[kind] = [row for row in self.history[kind] if row["step"] != record["step"]]
        self.history[kind].append(record)
        self.history[kind].sort(key=lambda row: row["step"])
        if self.writer is not None:
            write_scalars(self.writer, kind, record)

    def train(self, record):
        with evaluation_state():
            self._put("train", record)

    def development(self, step, score, best, metric, duration):
        with evaluation_state():
            self._put("dev", {"step": step, "score": score, "best": best, "metric": metric,
                              "by_dataset": self.last_dev_sources, "duration_s": duration,
                              "input_stats": self.last_dev_input_stats})

    def test(self, record, baseline):
        with evaluation_state():
            if self.baseline is None and baseline is not None and self.writer is not None:
                replay(self.writer, {"train": [], "dev": [], "test": []}, baseline)
            self.baseline = baseline
            first = not self.history["test"]
            self._put("test", record, persist=False)  # HeldoutMonitor commits JSONL itself.
            self.plot(record["step"], force=first)
            if self.writer is not None:
                self.writer.flush()

    def plot(self, step, force=False):
        if force or step % self.options["plot_every"] == 0:
            if self.last_plot_step != step or force:
                with evaluation_state():
                    if self.options["plots"]:
                        plot_history(self.root, self.history)
                    else:
                        export_csv(self.root, self.history)
                self.last_plot_step = step

    def __exit__(self, kind, value, traceback):
        try:
            with evaluation_state():
                if self.options["plots"]:
                    plot_history(self.root, self.history)
                else:
                    export_csv(self.root, self.history)
        finally:
            try:
                if self.writer is not None:
                    with evaluation_state():
                        self.writer.close()
            finally:
                self.lock.__exit__(kind, value, traceback)
