"""Rebuild plots and optional TensorBoard scalars from existing logs, without inference."""
from pathlib import Path

from depth_project.io import read_json
from explore_a.evaluation.state import evaluation_state
from explore_a.observability import settings
from explore_a.observability.history import FILES, load_history
from explore_a.observability.observer import run_lock, tensorboard_writer
from explore_a.observability.plots import plot_history
from explore_a.observability.progress import message
from explore_a.observability.scalars import replay


def export(config, name=None, tensorboard=False):
    parent = Path(config["paths"]["runs"])
    if name is not None and (Path(name).name != name or name in ("", ".", "..")):
        raise ValueError("--run must be one experiment folder name")
    roots = [parent / name] if name is not None else sorted(parent.glob("*"))
    count = 0
    for root in roots:
        if not root.is_dir() or not any((root / name).exists() for name in FILES.values()):
            continue
        with evaluation_state():
            history = load_history(root)
            if not any(history.values()):
                continue
            plot_history(root, history)
            if tensorboard:
                with run_lock(root):
                    path = root / "heldout_baseline.json"
                    baseline = read_json(path) if path.exists() else None
                    writer = tensorboard_writer(root, settings(config)["tb_flush_s"])
                    try:
                        replay(writer, history, baseline)
                    finally:
                        writer.close()
        count += 1
        message("Exported", root / "plots", "and TensorBoard" if tensorboard else "")
    if not count:
        raise ValueError("No completed scalar records to plot; no models or data were loaded")
