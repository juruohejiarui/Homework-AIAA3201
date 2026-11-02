"""Stable TensorBoard tags for training, development and observational held-out results."""
import math

from explore_a.observability.history import development


def scalars(kind, row):
    corrections = row.get("correction_metrics", {})
    if not corrections:
        corrections = (row.get("depth_metrics") or {}).get("correction_metrics", {})
    for source, metrics in corrections.items():
        for name, value in metrics.items():
            yield f"{kind}/{source}/correction/{name}", value
    for source, metrics in row.get("losses_by_dataset", {}).items():
        for name, value in metrics.items():
            yield f"{kind}/{source}/{name}", value
    for key, value in row.get("input_stats", {}).items():
        yield f"{kind}_inputs/{key}", value
    if kind == "train":
        for key in ("loss", "depth_loss", "blend_loss", "reset_loss", "correction_loss",
                    "global_correction_loss", "local_correction_loss", "aux_depth_ratio",
                    "temporal_loss", "temporal_coefficient",
                    "gauge_fallback_fraction", "gt_coverage",
                    "lr", "update_s", "pairs_per_s", "peak_allocated_gib", "microbatch"):
            if key in row:
                yield "train/" + key, row[key]
        summary = row.get("depth_metrics")
        if summary:
            for scope, metrics in [("macro", summary["macro_mean"]),
                                   *summary["by_dataset"].items()]:
                for name, value in metrics.items():
                    yield f"train/{scope}/{name}", value
    elif kind == "dev":
        metric = row["metric"]
        summary = row.get("depth_metrics")
        if summary:
            yield f"dev/selection/{metric}", row["score"]
            for scope, metrics in [("macro", summary["macro_mean"]),
                                   *summary["by_dataset"].items()]:
                for name, value in metrics.items():
                    yield f"dev/{scope}/{name}", value
        else:
            yield f"dev/macro/{metric}", row["score"]
        if "best" in row:
            yield f"dev/best/{metric}", row["best"]
        if "duration_s" in row:
            yield "time/dev_seconds", row["duration_s"]
        if not summary:
            for source, value in row.get("by_dataset", {}).items():
                yield f"dev/{source}/{metric}", value
    else:
        for scope, metrics in [("macro", row["macro_mean"]), *row["by_dataset"].items()]:
            for metric, value in metrics.items():
                yield f"test/{scope}/{metric}", value
        for scope, metrics in [("macro", row.get("delta_macro_mean", {})),
                               *row.get("delta_by_dataset", {}).items()]:
            for metric, value in metrics.items():
                yield f"test_delta/{scope}/{metric}", value
        for scope, counts in [("macro", row["counts"]), *row["dataset_counts"].items()]:
            for key, value in counts.items():
                yield f"test_counts/{scope}/{key}", value
        if "duration_s" in row:
            yield "time/test_seconds", row["duration_s"]


def write_scalars(writer, kind, row):
    for tag, value in scalars(kind, row):
        if math.isfinite(float(value)):
            writer.add_scalar(tag, value, int(row["step"]))


def replay(writer, history, baseline=None):
    for kind in ("train", "dev", "test"):
        rows = development(history) if kind == "dev" else history[kind]
        for row in rows:
            write_scalars(writer, kind, row)
    for tag, value in baseline_scalars(baseline):
        writer.add_scalar(tag, value, 0)


def baseline_scalars(baseline):
    if baseline:
        for scope, metrics in [("macro", baseline["macro_mean"]), *baseline["by_dataset"].items()]:
            for metric, value in metrics.items():
                yield f"baseline/{scope}/{metric}", value
