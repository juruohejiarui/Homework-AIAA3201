"""Shared training-dev/held-out collector; live weights never acquire gradients in evaluation."""
from collections import defaultdict
from contextlib import nullcontext
import copy
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

import numpy as np
import torch

from explore_a.data.reading import depth_range
from explore_a.evaluation.alignment import RelativeScores
from explore_a.evaluation.depth import MetricSummary, field_scores
from explore_a.evaluation.state import evaluation_state
from explore_a.training.correction_metrics import CorrectionMetrics
from explore_a.training.prediction import predict


def collect(model, dataset, config, device, occlusion, include_baseline=False, step=None,
            predict_fn=predict, progress=None):
    from explore_a.observability.progress import Progress
    scores, baseline = MetricSummary(), MetricSummary() if include_baseline else None
    corrections = CorrectionMetrics(config["training"]["correction_fraction"])
    aligned, aligned_base = RelativeScores(), RelativeScores()
    selection = defaultdict(list)
    phase = getattr(dataset, "phase", "dev")
    remote_fraction = config.get("evaluation_execution", {}).get("remote_cases", 0.)
    use_cases = (remote_fraction and config.get("execution", {}).get("backend") == "hybrid"
                 and config.get("remote_online", {}).get("enabled", False))
    if use_cases:
        iterator = routed(model, dataset, config, device, occlusion, step)
    else:
        def batches():
            for indices, data in dataset.iter_batches():
                with torch.autocast(device_type=device, enabled=device == "cuda" and config["training"]["amp"]):
                    output, _ = predict_fn(model, data, config, occlusion)
                yield [dataset.rows[i] for i in indices], data, output
        iterator = batches()
    try:
        runtime = getattr(dataset, "runtime", None)
        phase_scope = runtime.input_phase(phase) if use_cases and hasattr(runtime,"input_phase") else nullcontext()
        progress_scope = (nullcontext(progress) if progress is not None else
                          Progress(config, f"{phase.upper()} step {step}", len(dataset.rows)))
        with evaluation_state(model), phase_scope, progress_scope as progress:
            for rows, data, output in iterator:
                corrections.add(output, rows)
                for i, row in enumerate(rows):
                    gt, valid = data["gt"][i], data["gt_valid"][i].bool()
                    measured = field_scores(output["depth"][i], gt, valid, config, row["dataset"]) or {}
                    measured.update(aligned.add(row, output["depth"][i], gt, valid,
                                                depth_range(row["dataset"], config["data"])))
                    scores.add(row, measured or None, valid.sum())
                    if valid.any():
                        from explore_a.frontend import relative
                        from explore_a.evaluation.depth import frame_scores
                        selected = (measured["ssi_abs_rel"] if relative(config) else
                                    frame_scores(output["depth"][i], gt, valid,
                                                 all_metrics=False)["abs_rel"])
                        selection[row["dataset"], row["sequence"]].append(selected)
                    if baseline is not None:
                        base = data["current_depth"][i]
                        metrics = field_scores(base, gt, valid, config, row["dataset"]) or {}
                        metrics.update(aligned_base.add(row, base, gt, valid,
                                                        depth_range(row["dataset"], config["data"])))
                        baseline.add(row, metrics or None, valid.sum())
                progress.update(len(rows), dataset=rows[-1]["dataset"], sequence=rows[-1]["sequence"])
            for row, metrics, _ in aligned.finish():
                scores.extend_metrics(row, metrics)
            for row, metrics, _ in aligned_base.finish():
                if baseline is not None:
                    baseline.extend_metrics(row, metrics)
    finally:
        iterator.close()
        aligned.close()
        aligned_base.close()
    by_source = defaultdict(list)
    for (source, _), values in selection.items():
        by_source[source].append(float(np.mean(values)))
    if not by_source:
        raise ValueError("No valid evaluation supervision")
    result = scores.finish()
    result["selection_by_dataset"] = {k: float(np.mean(v)) for k,v in by_source.items()}
    result["selection_score"] = float(np.mean([np.mean(v) for v in by_source.values()]))
    result["correction_metrics"] = corrections.finish()
    if remote_fraction and hasattr(dataset, "last_evaluation_execution"):
        result["execution"] = dataset.last_evaluation_execution
    return result, baseline.finish() if baseline is not None else None


def routed(model, dataset, config, device, occlusion, step):
    """Use the standalone case worker with one immutable snapshot of the live head."""
    from explore_a.evaluation.session import identity, forward_frame
    from explore_a.evaluation.transport import RemotePair, case_route, assign_cases
    from explore_a.online.factory import make_runtime
    from explore_a.online.loading import OnlinePairs, read_pair
    from explore_a.online.pool import Workers
    from queue import Queue, Empty, Full
    from threading import Event
    cfg = copy.deepcopy(config)
    cfg["execution"]["routing"].setdefault("evaluation", {})["remote"] = 0.
    flow = dataset.flow
    fusion = {"GuidedUNet": "unet_guided", "FlowRefine": "flow_refine", "FinalFlow": "final_flow"}[type(model).__name__]
    task = {"flow": flow, "occlusion": occlusion, "fusion": fusion}
    with TemporaryDirectory(prefix="live-dev-snapshot-") as temp:
        path = Path(temp)/"checkpoint.pt"
        state = {"model": {k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                 "identity": {"experiment": task, "training": config["training"], "frontend": {"name":"da2", "kind":"relative_inverse"}},
                 "step": step, "best": float("inf")}
        torch.save(state, path)
        binding = {"entry": {"name": "live", "checkpoint": str(path)}, "config": config,
                   "state": state, "task": task, "checkpoint": path, "device": device, "model": model}
        peer, runtime = RemotePair([binding], config), make_runtime(cfg, [flow], occlusion == "mftiq", device)
        if hasattr(getattr(dataset,"runtime",None),"local"):
            runtime.local = dataset.runtime.local
        groups = defaultdict(list)
        for row in dataset.rows:
            groups[row["dataset"], row["sequence"], row["camera"], row.get("evaluation_clip", "full-sequence")].append(row)
        cases = assign_cases([{"scope": dataset.phase, "dataset": key[0], "key": identity(key), "rows": rows}
                              for key,rows in groups.items()], config)
        host_rows = [row for case in cases if case["owner"] == "host" for row in case["rows"]]
        remote_rows = [row for case in cases if case["owner"] == "dgx" for row in case["rows"]]
        data_source = OnlinePairs(cfg, host_rows, runtime, flow, occlusion == "mftiq", phase=dataset.phase)
        stream = data_source.raw_stream()
        ready, stopped = Queue(maxsize=2), Event()
        scheduler = Workers(1, "dev-dgx-owner")
        def produce_remote():
            with torch.inference_mode():
                for row in remote_rows:
                    if stopped.is_set():return
                    raw = read_pair(row, config["data"])
                    values = peer.submit(raw, False, True, ["live"], details=True).result()
                    while not stopped.is_set():
                        try:
                            ready.put((row, raw, values), timeout=.1)
                            break
                        except Full:
                            pass
        future = scheduler.submit(produce_remote)
        def merged():
            for index, raw in stream:
                yield host_rows[index], raw, forward_frame([binding], raw, runtime, details=True)
                while True:
                    try:yield ready.get_nowait()
                    except Empty:break
                if future.done():future.result()
            while not future.done() or not ready.empty():
                try:yield ready.get(timeout=.1)
                except Empty:pass
            future.result()
        merged_stream = merged()
        try:
            for row, raw, values in merged_stream:
                data = {k:v.to(device).unsqueeze(0) for k,v in raw["data"].items()}
                data["current_depth"] = torch.as_tensor(values["raw_canvas"], device=device)[None, None]
                output = {"depth": torch.as_tensor(values["live"], device=device)[None,None],
                          "prediction_kind":"relative_inverse", "correction":torch.zeros_like(data["current_depth"])}
                output.update({k.removeprefix("live/specific_out/"): torch.as_tensor(v, device=device)
                               for k,v in values.items() if k.startswith("live/specific_out/")})
                yield [row], data, output
        finally:
            stopped.set()
            # Abort remote receive before any scheduler/reader cleanup can wait for it.
            peer.close()
            runtime.close()
            merged_stream.close(); stream.close()
            scheduler.close()
            dataset.last_evaluation_execution = {"host_frames": len(host_rows), "dgx_frames": len(remote_rows),
                                                 "remote": dict(peer.counts), "host": runtime.statistics()}
