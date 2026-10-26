"""Nested terminal progress, periodic plain-text logs and progress-safe messages."""
import sys
import time

from tqdm.auto import tqdm

from explore_a.observability import settings

_ACTIVE = []


def message(*values):
    tqdm.write(" ".join(str(value) for value in values), file=sys.stderr)


def compact(value):
    return f"{value:.4g}" if isinstance(value, float) else str(value)


class Progress:
    def __init__(self, config, label, total, initial=0, unit="pair", stream=None):
        self.options, self.label = settings(config), label
        self.total, self.n, self.initial = total, initial, initial
        self.stream = sys.stderr if stream is None else stream
        self.bar, self.fields = None, {}
        self.started = self.last_report = time.monotonic()
        mode = self.options["progress"]
        self.enabled = mode != "off"
        self.terminal = mode == "bar" or (mode == "auto" and self.stream.isatty())
        self.unit = unit

    def __enter__(self):
        self.started = self.last_report = time.monotonic()
        if self.enabled:
            self._line("START")
            if self.terminal:
                self.bar = tqdm(total=self.total, initial=self.n, desc=self.label, unit=self.unit,
                                file=self.stream, position=len(_ACTIVE), dynamic_ncols=True,
                                mininterval=self.options["refresh_seconds"], leave=True)
            _ACTIVE.append(self)
        return self

    def set(self, **fields):
        changed = "phase" in fields and fields["phase"] != self.fields.get("phase")
        self.fields.update(fields)
        if self.bar is not None:
            self.bar.set_postfix({key: compact(value) for key, value in self.fields.items()},
                                 refresh=False)
            if changed:
                self.bar.refresh()

    def update(self, amount=1, **fields):
        self.set(**fields)
        self.n += amount
        if self.n > self.total:
            raise ValueError(f"Progress exceeded total: {self.label}")
        if self.bar is not None:
            self.bar.update(amount)
        elif (self.enabled and time.monotonic() - self.last_report >=
              self.options["non_tty_every_s"]):
            self._line("PROGRESS")

    def _line(self, event):
        elapsed = time.monotonic() - self.started
        done = self.n - self.initial
        speed = done / max(elapsed, 1e-9)
        eta = (self.total - self.n) / speed if speed else None
        fraction = self.n / max(self.total, 1) * 100
        postfix = " ".join(f"{key}={compact(value)}" for key, value in self.fields.items())
        line = (f"{event} {self.label}: {self.n}/{self.total} ({fraction:.1f}%) "
                f"elapsed={elapsed:.1f}s speed={speed:.3f} {self.unit}/s "
                f"ETA={'?' if eta is None else f'{eta:.1f}s'} {postfix}")
        if self.stream is sys.stderr:
            message(line)
        else:
            print(line, file=self.stream, flush=True)
        self.last_report = time.monotonic()

    def __exit__(self, kind, value, traceback):
        if self.bar is not None:
            self.bar.close()
        if self in _ACTIVE:
            _ACTIVE.remove(self)
        if self.enabled:
            self._line("FAILED" if kind else "END")
