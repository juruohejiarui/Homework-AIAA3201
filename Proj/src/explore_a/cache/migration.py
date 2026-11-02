"""Recoverable metadata-only recipe rename; prediction files are neither decoded nor rewritten."""
from contextlib import ExitStack
from pathlib import Path
import os

import yaml

from depth_project.io import read_json, write_json
from explore_a.cache.locks import process_lock
from explore_a.cache.paths import component
from explore_a.cache.settings import root_path


def assert_idle():
    """Inspect actual Python argv, avoiding shell command text and unrelated workloads."""
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            arguments = (process / "cmdline").read_bytes().split(b"\0")
            if not arguments or b"python" not in Path(os.fsdecode(arguments[0])).name.encode():
                continue
            decoded = [os.fsdecode(a) for a in arguments]
        except (OSError, ProcessLookupError):
            continue
        if "explore_a.cache.cli" in decoded and any(
                action in decoded for action in ("worker", "pipeline")):
            raise ValueError(f"Stop cache worker/coordinator before migration: PID {process.name}")
        if "explore_a" in decoded and any(action in decoded for action in
                                          ("train", "evaluate",
                                           "run-matrix", "evaluate-matrix", "all")):
            raise ValueError(f"Stop cache readers before migration: PID {process.name}")


def rewrite(value, old, new):
    if isinstance(value, dict):
        return {k: rewrite(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [rewrite(v, old, new) for v in value]
    if isinstance(value, str):
        if value == old:
            return new
        # Absolute cache references may point at either host's storage root.
        return value.replace("/explore_a/" + old + "/", "/explore_a/" + new + "/").replace(
            "/explore_a/" + old, "/explore_a/" + new) if "/explore_a/" + old in value else value
    return value


def migrate(base, previous, recipe):
    base = Path(base).expanduser()
    previous, recipe = component(previous), component(recipe)
    if previous == recipe:
        raise ValueError("Source and destination recipes must differ")
    assert_idle()
    with ExitStack() as stack:
        stack.enter_context(process_lock(base / "coordinator.lock", "Cache coordinator"))
        stack.enter_context(process_lock(base / "worker.lock", "GPU worker"))
        old, target = root_path(base, previous), root_path(base, recipe)
        if target.is_symlink():
            raise ValueError(f"Destination must be a real directory: {target}")
        if old.is_symlink():
            if old.resolve() != target.resolve() or not target.is_dir():
                raise ValueError(f"Conflicting compatibility link: {old}")
            if not (target / ".migration.json").exists():
                return {"root": str(target), "already_migrated": True}
        elif target.exists():
            if old.exists() or not (target / ".migration.json").exists():
                raise ValueError(f"Destination already exists: {target}")
        elif old.is_dir():
            write_json(old / ".migration.json", {"from": previous, "to": recipe})
            old.rename(target)
        else:
            raise FileNotFoundError(f"Legacy cache root is absent: {old}")
        if read_json(target / ".migration.json") != {"from": previous, "to": recipe}:
            raise ValueError("A different migration is pending")
        paths = list(target.glob("*.json")) + list(target.glob("*.yaml"))
        paths += list((target / "jobs").glob("*/*.json"))
        for path in paths:
            if path.name == ".migration.json":
                continue
            backup = target / ".migration" / "backups" / path.relative_to(target)
            if not backup.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                temporary = backup.with_name(backup.name + ".tmp")
                temporary.write_bytes(path.read_bytes())
                temporary.replace(backup)
            value = (read_json(backup) if path.suffix == ".json" else
                     yaml.safe_load(backup.read_text()))
            value = rewrite(value, previous, recipe)
            if path.suffix == ".json":
                write_json(path, value)
            else:
                from explore_a.cache.plan import write_yaml
                write_yaml(path, yaml.safe_dump(value, sort_keys=False))
        if not old.is_symlink():
            old.symlink_to(target.name, target_is_directory=True)
        write_json(target / ".migration" / "complete.json", {"from": previous, "to": recipe})
        (target / ".migration.json").unlink()
        return {"root": str(target), "compatibility_link": str(old),
                "metadata_files": len(paths), "predictions_rewritten": 0}
