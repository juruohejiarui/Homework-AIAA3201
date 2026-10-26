"""OS-backed process locks; a crashed process releases ownership without expiring jobs."""
from contextlib import contextmanager
import fcntl
from pathlib import Path


@contextmanager
def process_lock(path, description):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"{description} is already active: {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
