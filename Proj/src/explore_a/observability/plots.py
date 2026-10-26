"""Headless plots derived only from committed scalar logs; no model or dataset access."""
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from depth_project.io import read_json
from explore_a.evaluation.depth import DEPTH_METRICS
from explore_a.observability.history import development, load_history
from explore_a.observability.table import export_csv


def save(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        figure.tight_layout()
        figure.savefig(temporary, dpi=140)
        temporary.replace(path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)


def line(axis, rows, key, label, **kwargs):
    selected = [row for row in rows if key in row]
    if selected:
        axis.plot([row["step"] for row in selected], [row[key] for row in selected],
                  label=label, marker=".", markersize=3, **kwargs)
    axis.set_xlabel("Optimizer step")
    axis.grid(alpha=.2)



def depth_figure(root, rows, baseline, source, delta=False):
    figure, axes = plt.subplots(3, 3, figsize=(12, 9))
    scope = "macro_mean" if source == "macro" else "by_dataset"
    for axis, metric in zip(axes.flat, DEPTH_METRICS):
        points = []
        for row in rows:
            if delta:
                values = row.get("delta_macro_mean", {})
            else:
                values = row[scope] if source == "macro" else row[scope].get(source, {})
            if metric in values:
                points.append({"step": row["step"], "value": values[metric]})
        line(axis, points, "value", "test - Depth Pro" if delta else "held-out test")
        reference = baseline.get("macro_mean", {}) if source == "macro" else \
            baseline.get("by_dataset", {}).get(source, {})
        if delta:
            axis.axhline(0, color="gray", linestyle="--", label="no change")
        elif metric in reference:
            axis.axhline(reference[metric], color="gray", linestyle="--", label="Depth Pro")
        direction = " (higher better)" if metric.startswith("d") else " (lower better)"
        axis.set_title(metric + direction)
        axis.legend(fontsize=8)
    for axis in list(axes.flat)[len(DEPTH_METRICS):]:
        axis.set_visible(False)
    figure.suptitle(f"{root.name}: test {'delta' if delta else 'depth'} / {source} (monitor only)")
    save(figure, root / "plots" / f"test_{'delta' if delta else 'depth'}_{source}.png")


def plot_history(root, history=None):
    root = Path(root)
    history = load_history(root) if history is None else history
    export_csv(root, history)
    train, test, dev = history["train"], history["test"], development(history)
    student = root.name.startswith("occlusion-") or any(
        row.get("task") == "mftiq_soft_label_distillation" for row in test)
    if train:
        figure, axes = plt.subplots(2, 2, figsize=(11, 7))
        for axis, key, label in zip(axes.flat,
                                    ("loss", "lr", "pairs_per_s", "peak_allocated_gib"),
                                    ("Training loss", "Learning rate", "Pairs/s (update only)",
                                     "Peak CUDA allocated GiB")):
            line(axis, train, key, label)
            axis.set_title(label)
        figure.suptitle(root.name)
        save(figure, root / "plots/training.png")
    if dev:
        figure, axis = plt.subplots(figsize=(9, 5))
        line(axis, dev, "score", "dev macro")
        line(axis, dev, "best", "best dev", linestyle="--")
        sources = sorted({source for row in dev for source in row.get("by_dataset", {})})
        for source in sources:
            points = [{"step": row["step"], "value": row["by_dataset"][source]}
                      for row in dev if source in row.get("by_dataset", {})]
            line(axis, points, "value", source)
        axis.set_title(f"Development {dev[0]['metric']} (checkpoint selection)")
        axis.legend()
        save(figure, root / "plots/development.png")
    if student:
        figure, axis = plt.subplots(figsize=(9, 5))
        line(axis, train, "loss", "train soft-label BCE")
        line(axis, dev, "score", "dev soft-label BCE")
        points = [{"step": row["step"], "bce": row["macro_mean"]["heldout_bce"]}
                  for row in test if "heldout_bce" in row["macro_mean"]]
        line(axis, points, "bce", "test soft-label BCE (monitor only)")
        axis.set_title("MFTIQ pseudo-label distillation; not occlusion GT accuracy")
        if train or dev or points:
            axis.legend()
        save(figure, root / "plots/distillation.png")
    elif test:
        path = root / "heldout_baseline.json"
        baseline = read_json(path) if path.exists() else {}
        scopes = ["macro", *sorted({key for row in test for key in row["by_dataset"]})]
        for source in scopes:
            depth_figure(root, test, baseline, source)
        if any("delta_macro_mean" in row for row in test):
            depth_figure(root, test, baseline, "macro", delta=True)
