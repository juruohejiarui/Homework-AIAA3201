"""Persist test curves without influencing optimizers, schedules or best-checkpoint selection."""
import csv
import time
from pathlib import Path

from depth_project.io import read_json, write_json
from explore_a.config import bind_identity
from explore_a.observability.history import append_record, load_history
from explore_a.observability.progress import message
from explore_a.evaluation.heldout import collect_heldout


def scheduled(step, total, every):
    return every > 0 and (step == 0 or step == total or step % every == 0)


class HeldoutMonitor:
    def __init__(self, root, dataset, config, identity, occlusion, student=None,
                 distillation=False, device="cuda", observer=None):
        self.observer = observer
        self.root, self.dataset, self.config = Path(root), dataset, config
        self.occlusion, self.student = occlusion, student
        self.distillation, self.device = distillation, device
        bind_identity(self.root / "heldout_identity.json", {
            "input": identity["input"], "experiment": identity["experiment"],
            "teacher": identity.get("teacher"),
            "split": "test", "protocol": "metric depth, final-evaluation clipping, macro v1",
        })
        self.log = self.root / "heldout.jsonl"
        self.records = self._load_records()
        self.steps = {record["step"] for record in self.records}
        self.baseline_path = self.root / "heldout_baseline.json"
        self.baseline = read_json(self.baseline_path) if self.baseline_path.exists() else None
        if self.records:
            self._write_csv()  # Recover if interrupted after JSONL append but before CSV replace.

    def _load_records(self):
        records = load_history(self.root)["test"]
        if any(record["split"] != "test" for record in records):
            raise ValueError(f"Wrong split in {self.log}")
        return records

    def maybe_run(self, model, step):
        every = self.config["training"]["heldout_every"]
        if step in self.steps or not scheduled(step, self.config["training"]["steps"], every):
            return
        message(f"HELD-OUT TEST start: step {step}, full {len(self.dataset)} test pairs")
        began = time.monotonic()
        result, baseline = collect_heldout(
            model, self.dataset, self.config, self.device, self.occlusion, self.student,
            self.distillation, include_baseline=self.baseline is None, step=step,
        )
        if baseline is not None:
            self.baseline = baseline
            write_json(self.baseline_path, baseline)
        record = {"step": step, "split": "test", "selection_use": "monitor_only",
                  "duration_s": time.monotonic() - began,
                  "task": "mftiq_soft_label_distillation" if self.distillation else "metric_depth",
                  "counts_note": "valid pixels are correspondence-mask pixels" if self.distillation
                                 else "valid pixels are measured GT depth pixels",
                  **result}
        if self.baseline is not None and not self.distillation:
            record["delta_macro_mean"] = differences(result["macro_mean"],
                                                       self.baseline["macro_mean"])
            record["delta_by_dataset"] = {
                source: differences(metrics, self.baseline["by_dataset"][source])
                for source, metrics in result["by_dataset"].items()}
        append_record(self.log, record)
        self.records.append(record)
        self.steps.add(step)
        self._write_csv()
        if self.observer is not None:
            self.observer.test(record, self.baseline)
        message("HELD-OUT TEST (monitor only)", {
            "step": step, **result["macro_mean"],
            "delta_vs_depthpro": record.get("delta_macro_mean"),
            "valid_frames": result["counts"]["valid_frames"],
            "by_dataset": result["by_dataset"],
        })

    def _write_csv(self):
        rows = []
        for record in self.records:
            scopes = [("__macro__", record["macro_mean"], record["counts"],
                       record.get("delta_macro_mean", {}))]
            scopes.extend((source, metrics, record["dataset_counts"][source],
                           record.get("delta_by_dataset", {}).get(source, {}))
                          for source, metrics in record["by_dataset"].items())
            for source, metrics, counts, delta in scopes:
                rows.append({"step": record["step"], "split": "test", "task": record["task"],
                             "dataset": source, **counts, **metrics,
                             **{"delta_" + key: value for key, value in delta.items()}})
        fields = ["step", "split", "task", "dataset"]
        fields += sorted({key for row in rows for key in row} - set(fields))
        path = self.root / "heldout.csv"
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)


def differences(current, baseline):
    """Signed current-minus-baseline: negative depth errors / positive delta accuracy improve."""
    return {key: current[key] - baseline[key] for key in current}
