"""Observation artifacts, rollback views and terminal UI do not change training decisions."""
import copy
import csv
import io
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from depth_project.io import read_json
from explore_a.observability.export import export
from explore_a.observability.history import append_record, load_history, new_session
from explore_a.observability.observer import RunObserver, run_lock
from explore_a.observability.progress import Progress
from explore_a.training.engine import run_training
from test_explore_a_pipeline import online_experiment, fake_backbones as fake_backbones


def test_text_progress_initial_total_periodic_and_exception(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("explore_a.observability.progress.time.monotonic", lambda: clock[0])
    config = {"observability": {"progress": "auto", "non_tty_every_s": 30}}
    stream = io.StringIO()
    with Progress(config, "TRAIN", 5, initial=2, unit="step", stream=stream) as progress:
        clock[0] += 31
        progress.update(2, phase="train", loss=.1)
        clock[0] += 31
        progress.update(1)
    text = stream.getvalue()
    assert "START TRAIN: 2/5" in text and "PROGRESS TRAIN: 4/5" in text
    assert "END TRAIN: 5/5" in text and "ETA=" in text and "loss=0.1" in text
    assert "\r" not in text
    with pytest.raises(RuntimeError):
        with Progress(config, "TEST", 2, stream=stream):
            raise RuntimeError("interrupted")
    assert "FAILED TEST" in stream.getvalue()


def test_nested_terminal_bars_close_on_error():
    class Terminal(io.StringIO):
        def isatty(self):
            return True
    stream = Terminal()
    config = {"observability": {"progress": "auto"}}
    with pytest.raises(RuntimeError):
        with Progress(config, "TRAIN", 3, stream=stream) as train:
            train.update(1)
            with Progress(config, "DEV", 2, stream=stream) as dev:
                assert abs(dev.bar.pos) == 1
                dev.update(1)
                raise RuntimeError("stop")
    assert train.bar.disable and dev.bar.disable
    assert "\r" in stream.getvalue()


def test_session_rollback_preserves_raw_logs_and_drops_future_steps(tmp_path):
    train = tmp_path / "training.jsonl"
    for step in range(1, 5):
        append_record(train, {"step": step, "loss": step})
    original = train.read_bytes()
    new_session(tmp_path, 2)
    append_record(train, {"step": 3, "loss": 30})
    points = load_history(tmp_path)["train"]
    assert [p["step"] for p in points] == [1, 2, 3]
    assert points[-1]["loss"] == 30 and train.read_bytes().startswith(original)
    with train.open("a") as stream:
        stream.write('{"step":')
    new_session(tmp_path, 3)
    assert len(list(tmp_path.glob("training.jsonl.partial-*"))) == 1
    assert [p["step"] for p in load_history(tmp_path)["train"]] == [1, 2, 3]


def events(root):
    return EventAccumulator(str(root / "tensorboard"), size_guidance={"scalars": 0}).Reload()


def test_training_artifacts_rng_selection_and_event_resume(tmp_path, fake_backbones, capsys):
    config = online_experiment(tmp_path)
    states, counts = [], []
    depth = fake_backbones[0]
    for enabled in (False, True):
        current = copy.deepcopy(config)
        current["paths"]["runs"] += "-visible" if enabled else "-control"
        current["observability"].update(tensorboard=enabled, plots=enabled, plot_every=1,
                                         progress="text" if enabled else "off")
        depth.calls = 0
        run_training(current, "farneback", "rule", "cnn", device="cpu")
        counts.append(depth.calls)
        root = Path(current["paths"]["runs"]) / "farneback-rule-cnn"
        states.append(torch.load(root / "last.pt", weights_only=False))
        assert read_json(root / "complete.json")["steps"] == 2
        if enabled:
            log = events(root)
            tags = log.Tags()["scalars"]
            assert {"train/loss", "train/lr", "train/pairs_per_s", "dev/macro/abs_rel",
                    "test/macro/abs_rel", "test/tum/abs_rel", "baseline/macro/abs_rel",
                    "time/test_seconds", "test_counts/tum/valid_frames"} <= set(tags)
            assert [p.step for p in log.Scalars("test/macro/abs_rel")] == [0, 1, 2]
            assert [p.step for p in log.Scalars("train/loss")] == [1, 2]
            images = list((root / "plots").glob("*.png"))
            assert {p.name for p in images} >= {"training.png", "development.png",
                                               "test_depth_macro.png", "test_depth_tum.png",
                                               "test_delta_macro.png"}
            assert all(cv2.imread(str(path)) is not None for path in images)
            assert not plt.get_fignums()
            with (root / "scalars.csv").open() as stream:
                table = list(csv.DictReader(stream))
            curves = [row for row in table if row["tag"] == "test/macro/abs_rel"]
            assert [int(row["step"]) for row in curves] == [0, 1, 2]
            assert any(row["scope"] == "sequence" and row["sequence"] == "test" for row in table)
            assert float(curves[0]["value"]) == read_json(root / "heldout_baseline.json")[
                "macro_mean"]["abs_rel"]
            last_time = (root / "last.pt").stat().st_mtime_ns
            run_training(current, "farneback", "rule", "cnn", device="cpu")
            assert (root / "last.pt").stat().st_mtime_ns == last_time
            log = events(root)
            assert [p.step for p in log.Scalars("test/macro/abs_rel")] == [0, 1, 2]
            assert [p.step for p in log.Scalars("baseline/macro/abs_rel")] == [0]
            # Offline rendering and event rebuilding perform no further frozen inference.
            calls = depth.calls
            export(current, root.name, tensorboard=True)
            assert depth.calls == calls
    for key, value in states[0]["model"].items():
        torch.testing.assert_close(value, states[1]["model"][key], rtol=0, atol=0)
    torch.testing.assert_close(states[0]["torch_rng"], states[1]["torch_rng"])
    assert states[0]["best"] == states[1]["best"]
    assert states[0]["scheduler"] == states[1]["scheduler"]
    assert states[0]["identity"] == states[1]["identity"] and counts[0] == counts[1]
    text = capsys.readouterr().err
    assert "START TEST step 0" in text and "START TRAIN" in text and "START DEV step 1" in text


def test_student_logs_bce_only_and_writer_lock(tmp_path):
    config = {"observability": {"progress": "off", "plots": True, "tensorboard": True}}
    root = tmp_path / "occlusion-farneback"
    counts = {"total_frames": 2, "valid_frames": 2, "valid_pixels": 10}
    with RunObserver(root, config, 0) as observer:
        observer.train({"step": 1, "loss": .5, "lr": .001})
        observer.development(1, .4, .4, "bce", 1.0)
        record = {"step": 1, "split": "test", "task": "mftiq_soft_label_distillation",
                  "macro_mean": {"heldout_bce": .3}, "by_dataset": {"tum": {"heldout_bce": .3}},
                  "counts": counts, "dataset_counts": {"tum": counts}, "duration_s": 2}
        append_record(root / "heldout.jsonl", record)
        observer.test(record, None)
        with pytest.raises(ValueError, match="Active observer"):
            with run_lock(root):
                pass
    tags = events(root).Tags()["scalars"]
    assert "test/macro/heldout_bce" in tags and "dev/macro/bce" in tags
    assert not any("abs_rel" in tag or "occlusion_accuracy" in tag for tag in tags)
    assert (root / "plots/distillation.png").exists()
    with run_lock(root):
        pass
