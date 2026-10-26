"""Session-aware views retain raw logs but discard invalid tails after checkpoint rollback."""
import json
from pathlib import Path
import time
import uuid

FILES = {"train": "training.jsonl", "dev": "validation.jsonl", "test": "heldout.jsonl"}


def read_lines(path):
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    rows, offset = [], 0
    for number, line in enumerate(lines):
        offset += len(line)
        if not line.strip():
            continue
        try:
            rows.append((offset, json.loads(line)))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if number != len(lines) - 1:
                raise ValueError(f"Corrupt log before final record: {path}")
    return rows


def append_record(path, record):
    """Preserve a truncated last fragment before repairing the append boundary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        data = path.read_bytes()
        last = data.splitlines(keepends=True)[-1]
        try:
            json.loads(last)
        except (json.JSONDecodeError, UnicodeDecodeError):
            backup = path.with_name(path.name + f".partial-{time.time_ns()}")
            backup.write_bytes(last)
            with path.open("r+b") as stream:
                stream.truncate(len(data) - len(last))
        else:
            if not data.endswith(b"\n"):
                with path.open("ab") as stream:
                    stream.write(b"\n")
    with path.open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def new_session(root, step):
    root = Path(root)
    # Normalize append boundaries before capturing byte offsets for this session.
    for filename in FILES.values():
        path = root / filename
        if path.exists() and path.stat().st_size:
            data = path.read_bytes()
            try:
                json.loads(data.splitlines(keepends=True)[-1])
            except (json.JSONDecodeError, UnicodeDecodeError):
                last = data.splitlines(keepends=True)[-1]
                path.with_name(path.name + f".partial-{time.time_ns()}").write_bytes(last)
                with path.open("r+b") as stream:
                    stream.truncate(len(data) - len(last))
            else:
                if not data.endswith(b"\n"):
                    with path.open("ab") as stream:
                        stream.write(b"\n")
    session = {"id": uuid.uuid4().hex, "resume_step": step, "time": time.time(),
               "offsets": {key: (root / filename).stat().st_size
                           if (root / filename).exists() else 0 for key, filename in FILES.items()}}
    append_record(root / "observability/sessions.jsonl", session)
    return session


def load_history(root):
    root = Path(root)
    sessions = [row for _, row in read_lines(root / "observability/sessions.jsonl")]
    result = {}
    for key, filename in FILES.items():
        rows = read_lines(root / filename)
        effective, previous = {}, 0
        for session in sessions:
            boundary = session["offsets"][key]
            for end, row in rows:
                if previous < end <= boundary:
                    effective[int(row["step"])] = row
            effective = {step: row for step, row in effective.items()
                         if step <= session["resume_step"]}
            previous = boundary
        for end, row in rows:
            if end > previous:
                effective[int(row["step"])] = row
        result[key] = [effective[step] for step in sorted(effective)]
    return result


def development(history):
    if history["dev"]:
        return history["dev"]
    # Older online runs stored only the overall development score in training.jsonl.
    return [{"step": row["step"], "score": row[key], "metric": key.removeprefix("dev_"),
             "by_dataset": {}} for row in history["train"]
            for key in ("dev_abs_rel", "dev_bce") if key in row]

