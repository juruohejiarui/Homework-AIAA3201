"""Case-major evaluation: one frozen input graph, many serial heads and one baseline."""
from collections import defaultdict
from contextlib import ExitStack
import copy
import csv
import hashlib
import json
import re
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
import time

import cv2
import numpy as np
import torch

from depth_project.io import write_json
from depth_project.models.api import ModelInput
from depth_project.paths import PROJECT
from explore_a.config import load_config
from explore_a.data.prepare import load_manifest
from explore_a.data.reading import depth_range, fit_image
from explore_a.evaluation.accumulator import average
from explore_a.evaluation.alignment import RelativeScores
from explore_a.evaluation.sintel_ac import PROTOCOL as SINTEL_AC, IMAGE_METRICS, prepare_manifest, image_scores
from explore_a.evaluation.state import evaluation_state
from explore_a.online.factory import make_runtime
from explore_a.online.loading import read_pair
from explore_a.training.checkpoint import load_model_state
from explore_a.training.prediction import make_fusion, fusion_geometry, predict

VERSION = "shared-case-v2-ac-sintel"


def commit(path, value):
    path = Path(path)
    pending = path.with_name(path.name+".part")
    write_json(pending, value)
    pending.replace(path)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def weights_hash(path):
    path = Path(path)
    return (identity({str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob("*")) if p.is_file()})
            if path.is_dir() else file_hash(path))


def frozen_identity(bindings):
    required = {"da2"}
    if any(b.get("task", {}).get("flow") == "raft" for b in bindings):
        required.add("raft")
    if any(b.get("task", {}).get("occlusion") == "mftiq" for b in bindings):
        required.update(("mftiq", "resnet", "dinov2"))
    return {k: file_hash(bindings[0]["config"]["weights"][k]) for k in sorted(required)}


def implementation_hash(binding):
    if "task" in binding:
        names = {"unet_guided": "guided", "flow_refine": "flow_refine", "final_flow": "final_flow"}
        folder = PROJECT/"src/explore_a/models"
        return identity({name: file_hash(folder/f"{name}.py") for name in
                         (names[binding["task"]["fusion"]], "blocks", "inputs", "correction", "correlation")})
    import inspect
    from depth_project.models import adapter_type
    adapter = adapter_type(binding["entry"]["adapter"])
    paths = [Path(inspect.getfile(adapter))]
    paths.extend(p for root in getattr(adapter, "runtime_paths", ()) for p in (PROJECT/root).rglob("*.py"))
    return identity({str(p): file_hash(p) for p in sorted(set(paths))})


def entries(path, selected=None):
    values = json.loads(Path(path).read_text())
    names = [v["name"] for v in values]
    if len(set(names)) != len(names) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", n)
                                           or n in ("raw_canvas", "raw_native") for n in names):
        raise ValueError("Model names must be unique safe directory names")
    if any(v.get("kind", "fusion") not in ("fusion", "frozen", "ensemble") for v in values):
        raise ValueError("Model kind must be fusion, frozen or ensemble")
    if selected and set(selected) - set(names):
        raise ValueError("Unknown models: " + str(set(selected) - set(names)))
    return [v for v in values if not selected or v["name"] in selected]


def bind(entry, snapshots, device="cuda"):
    entry = copy.deepcopy(entry)
    config = load_config(PROJECT / entry.get("config", "configs/explore_a/mixed_h5_da2.yaml"))
    config["evaluation_dev_fraction"] = config["training"]["dev_sample_fraction"]
    value = {"entry": entry, "config": config, "model": None, "device": device}
    if entry.get("kind", "fusion") == "fusion":
        source = PROJECT / entry["checkpoint"]
        folder = snapshots / entry["name"]
        target = folder / "checkpoint.pt"
        receipt = folder / "source.json"
        if receipt.exists() and json.loads(receipt.read_text())["entry"] != entry:
            raise ValueError("Changed registry entry; choose a new model name/output")
        if not target.exists():
            state = torch.load(source, map_location="cpu", weights_only=False)
            state = {k: state[k] for k in ("model", "identity", "step", "best")}
            folder.mkdir(parents=True, exist_ok=True)
            pending = target.with_suffix(".part")
            torch.save(state, pending)
            pending.replace(target)
            write_json(receipt, {"entry": entry, "source_sha256": file_hash(source)})
        state = torch.load(target, map_location="cpu", weights_only=False)
        from explore_a.training.identity import check_frontend
        check_frontend(state["identity"], config)
        config["training"].update(state["identity"]["training"])
        value.update(state=state, task=state["identity"]["experiment"], checkpoint=target)
    else:
        weights = PROJECT / entry["weights"]
        if not weights.exists():
            raise FileNotFoundError(weights)
        value["weights"] = weights
    return value


def head(binding):
    if binding["model"] is None:
        model = make_fusion(binding["task"]["fusion"], binding["config"])
        load_model_state(model, binding["state"]["model"])
        binding["model"] = model.to(binding["device"]).eval().requires_grad_(False)
    return binding["model"]


def reference(binding):
    if binding["model"] is None:
        from depth_project.models import load_model
        e = binding["entry"]
        binding["model"] = load_model(e["adapter"], binding["weights"],
                                      **e.get("load_param", {}))
        model = getattr(binding["model"], "model", binding["model"])
        if isinstance(model, torch.nn.Module):
            model.eval().requires_grad_(False)
    return binding["model"]


def capability(binding):
    e = binding["entry"]
    if "capability" in e:
        return e["capability"]
    if ":" in e.get("adapter", ""):
        from depth_project.models.api import resolve_adapter
        return resolve_adapter(e["adapter"]).capability
    return "sequence" if e.get("adapter") == "vda" else "image"


def compatible(bindings):
    """Never silently use the first model's canvas/weights for an incompatible model."""
    def signature(binding):
        c = binding["config"]
        return {"data": c["data"], "geometry": c["geometry"], "frontend": c["frontend"],
                "dev_fraction": c.get("evaluation_dev_fraction", c["training"]["dev_sample_fraction"]),
                "raft_bidirectional_batch": c["online"]["raft_bidirectional_batch"],
                "weight": file_hash(c["weights"]["da2"])}
    if len({identity(signature(b)) for b in bindings}) > 1:
        raise ValueError("This session requires matching data/geometry/frontend; use separate outputs")


def forward_frame(bindings, raw, runtime, native=False, history=True, details=False):
    """Shared differentiable forward is also used by training through predict()."""
    config = bindings[0]["config"]
    base = runtime.depth_pair(raw)
    values = {"raw_canvas": base["current_depth"][0].detach().float().cpu().numpy()}
    if native:
        values["raw_native"] = raw["native_predictions"]["current"]
    prepared, geometries = {}, {}
    for b in bindings:
        name = b["entry"]["name"]
        if b["entry"].get("kind", "fusion") != "fusion":
            continue
        if not history:
            values[name] = values["raw_canvas"]
            continue
        task, c = b["task"], b["config"]
        key = task["flow"]
        if key not in prepared:
            teacher = any(other.get("task", {}).get("flow") == key
                          and other["task"]["occlusion"] == "mftiq" for other in bindings)
            data = runtime.add_flow(raw, base, task["flow"], teacher)
            prepared[key] = {k: v.unsqueeze(0) for k, v in data.items()}
            geometries[key] = fusion_geometry(prepared[key], c)
        with torch.autocast(device_type=b["device"], enabled=b["device"] == "cuda"
                            and c["training"]["amp"]):
            result, _ = predict(head(b), prepared[key], c, task["occlusion"], geometries[key])
        values[name] = result["depth"][0, 0].detach().float().cpu().numpy()
        if details:
            for k,v in result.items():
                if isinstance(v, torch.Tensor) and k != "depth":
                    values[name + "/specific_out/" + k] = v.detach().float().cpu().numpy()
    return values


class CaseScores:
    """GT is read once per input; spool valid arrays, not complete recordings in RAM."""
    def __init__(self, baseline_names=(), baseline_stats=None, sintel_ac=False):
        self.aligned, self.records = {}, defaultdict(list)
        self.baseline_names = set(baseline_names)
        self.baseline_stats = baseline_stats
        self.sintel_ac = sintel_ac

    def add(self, name, row, prediction, truth, valid, limits, representation="relative_inverse"):
        from depth_project.evaluation.metrics import depth_metrics
        p = np.asarray(prediction, np.float32)
        if p.ndim != 2:
            raise ValueError("Each reference prediction must be an HW depth field")
        if p.shape != truth.shape:
            p = cv2.resize(p, (truth.shape[1], truth.shape[0]))
        # Letterboxing inserts zeros outside the image. Only measured GT pixels
        # enter metric scoring; these padding values are not model predictions.
        nonfinite = int((~np.isfinite(p)).sum())
        scored = (np.isfinite(truth) & (truth > 0)) if self.sintel_ac else valid
        nonpositive = int((p[scored] <= 0).sum()) if representation == "metric_m" else 0
        if nonfinite or nonpositive:
            raise ValueError(f"Invalid prediction {name}/{row['id']}: "
                             f"nonfinite={nonfinite}, nonpositive_scored_pixels={nonpositive}")
        if representation == "metric_m":
            if valid.any():
                self.records[name].append({"metric_" + k: v for k, v in
                    depth_metrics(np.clip(p[valid], *limits), truth[valid]).items()})
            p = 1 / np.maximum(p, 1e-6)
        if name not in self.aligned:
            self.aligned[name] = RelativeScores()
        scorer = self.aligned[name]
        measured = scorer.add(row, np.maximum(p, 1e-3) if self.sintel_ac else p, truth, valid, limits)
        if self.sintel_ac:
            # All positive GT pixels and all frames enter the AC image metric,
            # even if this frame has no valid pixels under the video mask.
            measured = {k: v for k, v in measured.items() if k not in IMAGE_METRICS}
            measured.update(image_scores(p, truth))
        self.records[name].append(measured)

    def finish(self):
        try:
            anchor = self.aligned.get("raw_canvas")
            for name, scorer in self.aligned.items():
                fixed = ((anchor.stats if anchor is not None else self.baseline_stats)
                         if name in self.baseline_names else None)
                for row, measured, _ in scorer.finish(fixed):
                    self.records[name].append(measured)
            return {name: average(rows) for name, rows in self.records.items()}
        finally:
            self.close()

    def close(self):
        for scorer in self.aligned.values():
            scorer.close()


def cohorts(scope, config, benchmark_root, tum_root, setting="long", datasets=None, full_dev=False):
    if scope in ("test", "dev"):
        from explore_a.training.dev_subset import select_dev
        rows = [r for r in load_manifest(config)["samples"] if r["split"] == scope]
        if scope == "dev" and not full_dev:
            rows = select_dev(rows, config.get("evaluation_dev_fraction", config["training"]["dev_sample_fraction"]),
                              config["data"]["seed"])
        groups = defaultdict(list)
        for row in rows:
            groups[row["dataset"], row["sequence"], row["camera"],
                   row.get("evaluation_clip", "full-sequence")].append(row)
        for key, selected in groups.items():
            yield {"dataset": key[0], "sequence": key[1], "camera": key[2],
                   "key": identity(key), "rows": selected, "scope": scope}
        return
    from depth_project.data.protocol import select_sequences
    from explore_a.evaluation.part2 import MANIFESTS, benchmark_root_for
    from explore_a.evaluation.part2_protocol import benchmark_path
    datasets = ["tum-validation"] if scope == "tum-validation" else (datasets or ["kitti", "sintel", "bonn"])
    for ds in datasets:
        root = benchmark_root_for(ds, benchmark_root, tum_root)
        manifest = json.loads((root / MANIFESTS[ds]).read_text())
        prepare_manifest(manifest, root)
        selected = (manifest["sequences"] if ds == "sintel" else
                    select_sequences(manifest, "full" if scope == "tum-validation" else setting))
        for seq in selected:
            for f in seq["frames"]:
                for k in ("image", "gt"):
                    f[k] = benchmark_path(f[k], root)
            yield {"dataset": ds, "sequence": seq["name"], "camera": "video",
                   "key": identity([ds, seq["name"], setting]), "frames": seq["frames"],
                   "setting": "full" if scope == "tum-validation" else setting,
                   "protocol": manifest["config"], "scope": scope}


def samples(case, config):
    if "rows" in case:
        for row in case["rows"]:
            raw = read_pair(row, config["data"])
            yield raw, raw["data"]["gt"][0].numpy(), raw["data"]["gt_valid"][0].numpy(), True
        return
    from depth_project.evaluation.official import load_evaluator
    from explore_a.evaluation.part2 import raw_input, input_rgb
    evaluator = load_evaluator(device="cpu")
    previous = None
    frames, p = case["frames"], case["protocol"]
    for i, frame in enumerate(frames):
        now = input_rgb(frame, case["dataset"], "long", p["crop"])
        history = previous is not None and not frame.get("reset_history", False)
        before = previous if history else now
        raw = raw_input(before, now, case["dataset"], case["sequence"], str(max(0, i-1)), str(i), config)
        raw["row"].update(previous_rgb=frames[i-1 if history else i]["image"],
                          current_rgb=frame["image"],
                          input_transform={"scope": "part2", "dataset": case["dataset"],
                                           "crop": list(p["crop"]), "rgb": "official-crop-even-v1"})
        top, bottom, left, right = p["crop"]
        gt = evaluator.get_gt(frame["gt"], p["factor"], None)[top:bottom, left:right]
        valid = np.isfinite(gt) & (gt > 1e-3) & (gt < p["max_depth"])
        yield raw, gt, valid, history
        previous = now


def evaluate_case(case, bindings, output, runtime, device="cuda", remote=None,
                  ensembles=(), requested=None):
    from explore_a.evaluation.part2_protocol import restore_canvas
    c = bindings[0]["config"]
    frozen = [b for b in bindings if b["entry"].get("kind", "fusion") == "frozen"]
    expected = {"raw_canvas", *(requested if requested is not None else
                               [b["entry"]["name"] for b in bindings])}
    native = "frames" in case
    from explore_a.evaluation.transport import case_route
    remote_case = remote is not None and case_route(case, c) == "dgx"
    if native:
        expected.add("raw_native")
    folder = output / case["scope"] / case["dataset"] / case["key"]
    receipts = {name: folder / name / "metrics.json" for name in expected}
    # Each model is a commit unit. A failed model never invalidates completed siblings.
    completed = {n: json.loads(p.read_text()) for n, p in receipts.items() if p.is_file()}
    signature = identity(case)
    if any(r.get("cohort_sha256") != signature for r in completed.values()):
        raise ValueError("Completed case has different inputs/protocol")
    if len(completed) == len(expected):
        return list(completed.values())
    from explore_a.evaluation.ensemble import plan, EnsembleFields
    pending_ensembles = [e["name"] for e in ensembles if e["name"] in expected - completed.keys()]
    needed, definitions, _ = plan([b["entry"] for b in bindings] + list(ensembles),
                                  pending_ensembles)
    dependencies = {e["name"] for e in needed}
    active = [b for b in bindings if b["entry"]["name"] in dependencies
              or b["entry"]["name"] in expected - completed.keys()]
    baseline_stats = {tuple(item["key"]): np.array(item["moments"], np.float64) for item in
                      completed.get("raw_canvas", {}).get("alignment_stats", [])}
    ac = case.get("protocol", {}).get("image_scoring") == SINTEL_AC
    fixed_names = {b["entry"]["name"] for b in bindings
                   if "task" in b or b["entry"].get("reuse_frontend")}
    for definition in definitions:
        if set(definition["members"]) <= fixed_names:
            fixed_names.add(definition["name"])
    score = CaseScores({"raw_canvas", "raw_native", *fixed_names}, baseline_stats, sintel_ac=ac)
    began, count = time.monotonic(), 0
    flows = {b["task"]["flow"]: any(other.get("task", {}).get("flow") == b["task"]["flow"]
                                   and other["task"]["occlusion"] == "mftiq" for other in active)
             for b in active if "task" in b}
    # Bounded RGB spool supports sequence references without retaining an entire recording.
    with TemporaryDirectory(prefix="case-rgb-") as temp:
        sequence_refs = [b for b in frozen if b in active and not b["entry"].get("reuse_frontend")
                         and capability(b) == "sequence"]
        ensemble_fields = EnsembleFields(temp, definitions, deferred=bool(sequence_refs))

        def score_ensembles(index, row, gt, valid, limits):
            for name, (value, representation) in ensemble_fields.predictions(
                    index, pending_ensembles).items():
                if native and representation == "relative_inverse" and not ac:
                    value = np.maximum(value, 1e-3)
                score.add(name, row, value, gt, valid, limits, representation)

        try:
            for i, (raw, gt, valid, history) in enumerate(samples(case, c)):
                if remote_case:
                    values = remote.submit(raw, native, history,
                                           [b["entry"]["name"] for b in active]).result()
                else:
                    raw["frozen_ticket"] = runtime.prepare(raw["row"], i, flows, phase=case["scope"], native=native)
                    baseline = {"config": c, "entry": {"kind": "frozen", "name": "baseline"}}
                    values = forward_frame(active or [baseline], raw, runtime, native, history)
                row = {**raw["row"], "dataset": case["dataset"], "sequence": case["sequence"],
                       "camera": case["camera"], "evaluation_clip": case["key"]}
                limits = depth_range(case["dataset"], c["data"]) if not native else (1e-3, case["protocol"]["max_depth"])
                rgb = raw["images"][False]
                representations = {}
                for b in frozen:
                    name = b["entry"]["name"]
                    if b not in active or b in sequence_refs:
                        continue
                    e = b["entry"]
                    if e.get("reuse_frontend", False):
                        if e["adapter"] != "da2" or e.get("load_param", {}).get("fp32", False):
                            raise ValueError("Frontend alias must use matching DA2 FP16")
                        values[name] = values["raw_native" if native else "raw_canvas"]
                    else:
                        request = ModelInput(rgb=[rgb], frame_ids=(row["id"],), specific_in=e.get("specific_in", {}))
                        result = (remote.reference(b, request) if remote_case else
                                  reference(b).forward(request, **e.get("specific_param", {})))
                        value = result.numpy()[0]
                        values[name] = value if native else fit_image(value, raw["canvas"])[0]
                        representations[name] = result.representation
                        if remote_case:
                            # Image references have already been copied out of their temporary map.
                            values[name] = values[name].copy()
                            del result
                            remote.release_references()
                for name, value in values.items():
                    if native and name != "raw_native" and (name == "raw_canvas" or
                            any(b["entry"]["name"] == name and "task" in b for b in active)):
                        value = restore_canvas(value, rgb.shape, raw["canvas"])
                    elif not native and name == "raw_native":
                        continue
                    representation = representations.get(name, "relative_inverse")
                    ensemble_fields.store(name, i, value, representation, gt.shape)
                    if name in completed or name not in expected:
                        continue
                    if native and representation == "relative_inverse" and not ac:
                        value = np.maximum(value, 1e-3)
                    score.add(name, row, value, gt, valid, limits, representation)
                if not sequence_refs:
                    score_ensembles(i, row, gt, valid, limits)
                if sequence_refs:
                    np.save(Path(temp) / f"{i}.npy", rgb)
                    np.save(Path(temp) / f"gt-{i}.npy", gt)
                    np.save(Path(temp) / f"valid-{i}.npy", valid)
                    write_json(Path(temp) / f"row-{i}.json", row)
                count += 1
            for b in sequence_refs:
                frames = np.load(Path(temp) / "0.npy", mmap_mode="r")
                mapped = np.lib.format.open_memmap(Path(temp)/"video.npy", mode="w+", dtype=np.uint8,
                                                   shape=(count, *frames.shape))
                for i in range(count):
                    mapped[i] = np.load(Path(temp) / f"{i}.npy")
                e = b["entry"]
                request = ModelInput(rgb=mapped, specific_in=e.get("specific_in", {}))
                result = (remote.reference(b, request) if remote_case else
                          reference(b).forward(request, **e.get("specific_param", {})))
                predictions = result.numpy()
                for i in range(count):
                    gt = np.load(Path(temp) / f"gt-{i}.npy")
                    valid = np.load(Path(temp) / f"valid-{i}.npy")
                    value = predictions[i]
                    if not native:
                        canvas = c["data"]["canvas_kitti" if case["dataset"] == "kitti" else "canvas_indoor"]
                        value = fit_image(value, canvas)[0]
                    row = json.loads((Path(temp) / f"row-{i}.json").read_text())
                    limits = (1e-3, case["protocol"]["max_depth"]) if native else depth_range(case["dataset"], c["data"])
                    ensemble_fields.store(e["name"], i, value, result.representation, gt.shape)
                    if e["name"] in expected and e["name"] not in completed:
                        score.add(e["name"], row, value, gt, valid, limits, result.representation)
                del predictions, mapped
                if remote_case:
                    del result
                    remote.release_references()
            if sequence_refs and pending_ensembles:
                for i in range(count):
                    gt = np.load(Path(temp) / f"gt-{i}.npy")
                    valid = np.load(Path(temp) / f"valid-{i}.npy")
                    row = json.loads((Path(temp) / f"row-{i}.json").read_text())
                    limits = ((1e-3, case["protocol"]["max_depth"]) if native else
                              depth_range(case["dataset"], c["data"]))
                    score_ensembles(i, row, gt, valid, limits)
            for name, metrics in score.finish().items():
                record = {"scope": case["scope"], "dataset": case["dataset"],
                          "sequence": case["sequence"], "camera": case["camera"],
                          "case": case["key"], "model": name, "frames": count, "metrics": metrics,
                          "wall_s": time.monotonic()-began, "execution": runtime.statistics(),
                          "case_owner": "dgx" if remote_case else "host-cache",
                          "remote": getattr(remote, "provenance", None)}
                record["scoring_protocol"] = SINTEL_AC if ac else {"id": "image-video-depth-v1"}
                record["cohort_sha256"] = signature
                if name in pending_ensembles:
                    record["ensemble"] = next(e for e in definitions if e["name"] == name)
                if name in score.aligned:
                    record["alignment_stats"] = [{"key": list(key), "moments": value.tolist()}
                                                  for key,value in score.aligned[name].stats.items()]
                record["metric_counts"] = {k: sum(k in r for r in score.records[name]) for k in metrics}
                commit(receipts[name], record)
                completed[name] = record
        finally:
            score.close()
    print("CASE COMPLETE", case["scope"], case["dataset"], case["sequence"], count, flush=True)
    return list(completed.values())


def summarize(output):
    output = Path(output)
    if json.loads((output/"session.json").read_text()).get("schema") != VERSION:
        raise ValueError("Obsolete evaluation results; run evaluate with a new output directory")
    groups = defaultdict(list)
    for path in output.glob("*/*/*/*/metrics.json"):
        r = json.loads(path.read_text())
        if r["scope"] == "benchmark" and r["dataset"] == "sintel" and r.get("scoring_protocol") != SINTEL_AC:
            raise ValueError("Sintel results do not use the adopted AC protocol")
        groups[r["scope"], r["dataset"], r["model"], r["sequence"]].append(r)
    sources = defaultdict(list)
    for (scope, dataset, model, _), records in groups.items():
        keys = set().union(*(r["metrics"] for r in records))
        metrics = {k: sum(r["metrics"][k]*r["metric_counts"][k] for r in records if k in r["metrics"])
                   /sum(r["metric_counts"][k] for r in records if k in r["metrics"]) for k in keys}
        sources[scope, dataset, model].append(metrics)
    rows = [{"scope": s, "dataset": d, "model": m, **average(v)} for (s,d,m),v in sorted(sources.items())]
    for row in rows:
        ac = row["scope"] == "benchmark" and row["dataset"] == "sintel"
        row["image_standard"] = SINTEL_AC["id"] if ac else "image-video-depth-v1"
        if ac:
            records = [r for (s,d,m,_), group in groups.items() for r in group
                       if (s,d,m) == (row["scope"],row["dataset"],row["model"])]
            for key in IMAGE_METRICS:
                count = sum(r["metric_counts"].get(key, 0) for r in records)
                if count:
                    row[key] = sum(r["metrics"][key]*r["metric_counts"][key]
                                   for r in records if key in r["metrics"])/count
            row["image_frames"] = sum(r["metric_counts"].get("absrel_img", 0) for r in records)
    write_json(output/"comparison.json", rows)
    write_json(output/"scoring_protocols.json", {"sintel": SINTEL_AC})
    keys = ["scope", "dataset", "model", "image_standard", "image_frames", "absrel_img", "absrel_video", "delta1_img", "delta1_video"]
    lines = ["# Shared-input model comparison", "", "Public baselines are evaluated once.", "",
             "Sintel: final, all 1064 frames. Image metrics use AC (GT cap 72 m, prediction cap 80 m, "
             "per-image inverse alignment, depth metrics, frame mean). Video metrics use the 70 m mask "
             "and one alignment per sequence, then sequence macro. Other datasets retain their existing protocol.", "",
             "| " + " | ".join(keys) + " |", "| " + " | ".join(["---"]*len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(f"{row[k]:.9f}" if isinstance(row.get(k), (float,int))
                                       else str(row.get(k,"—")) for k in keys) + " |")
    (output/"comparison.md").write_text("\n".join(lines)+"\n")
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else keys
    with (output/"comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(rows)
    return rows


def run(registry, output, scopes, benchmark_root, tum_root, selected=None, device="cuda",
        setting="long", datasets=None, full_dev=False):
    output = Path(output)
    if (output/"comparison.json").exists() and not (output/"session.json").exists():
        raise ValueError("Old evaluation layout: choose a new --output")
    from explore_a.evaluation.ensemble import plan
    values, ensembles, requested = plan(entries(registry), selected)
    bindings = [bind(e, output/"models", device) for e in values]
    if not bindings:
        raise ValueError("Empty model registry")
    compatible(bindings)
    from explore_a.online.routing import profiles
    from depth_project.models.provenance import shared_signature
    numeric = {"geometry": bindings[0]["config"]["geometry"], "frontend": bindings[0]["config"]["frontend"],
               "dev_fraction": bindings[0]["config"]["evaluation_dev_fraction"],
               "raft_bidirectional_batch": bindings[0]["config"]["online"]["raft_bidirectional_batch"],
               "routing": profiles(bindings[0]["config"])["evaluation"],
               "execution": bindings[0]["config"].get("evaluation_execution", {}),
               "weights": frozen_identity(bindings), "source": shared_signature()}
    protocol = {"schema": VERSION, "numeric": numeric, "models": [{"entry": b["entry"], "checkpoint_sha256":
                  file_hash(b["checkpoint"]) if "checkpoint" in b else weights_hash(b["weights"]),
                  "implementation_sha256": implementation_hash(b)} for b in bindings],
                "data": bindings[0]["config"]["data"], "setting": setting, "full_dev": full_dev,
                "manifest_sha256": file_hash(bindings[0]["config"]["paths"]["manifest"])}
    from explore_a.evaluation import ensemble
    for entry in ensembles:
        members = {m["entry"]["name"]: m for m in protocol["models"]}
        protocol["models"].append({"entry": entry,
            "checkpoint_sha256": identity([members[name] for name in entry["members"]]),
            "implementation_sha256": file_hash(Path(ensemble.__file__))})
    path = output/"session.json"
    if path.exists() and json.loads(path.read_text()).get("schema") != VERSION:
        raise ValueError("Obsolete evaluation output; choose a new --output directory (v2 uses AC Sintel)")
    if path.exists() and json.loads(path.read_text()) != protocol:
        old = json.loads(path.read_text())
        old_numeric = copy.deepcopy(old.get("numeric", {}))
        old_weights = old_numeric.pop("weights", {})
        new_numeric = {k: v for k,v in numeric.items() if k != "weights"}
        if old_numeric != new_numeric or any(k in numeric["weights"] and numeric["weights"][k] != v
                                              for k,v in old_weights.items()):
            raise ValueError("Shared inference protocol/weights changed")
        protocol["numeric"]["weights"] = {**old_weights, **numeric["weights"]}
        old["numeric"] = protocol["numeric"]
        if any(old.get(k) != protocol[k] for k in ("schema", "numeric", "data", "setting", "full_dev", "manifest_sha256")):
            raise ValueError("Evaluation protocol changed")
        known = {m["entry"]["name"]: m for m in old["models"]}
        if any(known.get(m["entry"]["name"], m) != m for m in protocol["models"]):
            raise ValueError("Previously evaluated model changed")
        known.update({m["entry"]["name"]: m for m in protocol["models"]})
        protocol["models"] = [known[k] for k in sorted(known)]
    commit(path, protocol)
    config = bindings[0]["config"]
    flows = sorted({b["task"]["flow"] for b in bindings if "task" in b}) or ["farneback"]
    teacher = any(b.get("task",{}).get("occlusion") == "mftiq" for b in bindings)
    with ExitStack() as stack:
        from explore_a.online.pool import Workers
        host_config = copy.deepcopy(config)
        # Full-case remote fraction is independent of frozen-component training routing.
        host_config.setdefault("execution", {}).setdefault("routing", {}).setdefault("evaluation", {})["remote"] = 0.
        runtime = make_runtime(host_config, flows, teacher, device)
        stack.callback(runtime.close)
        remote = None
        if (config.get("execution", {}).get("backend") == "hybrid"
                and config.get("remote_online", {}).get("enabled", False)
                and config.get("evaluation_execution", {}).get("remote_cases", 0) > 0):
            from explore_a.evaluation.transport import RemotePair
            remote = RemotePair(bindings, config)
            stack.callback(remote.close)
        scheduler = Workers(1, "dgx-case-owner")
        stack.callback(scheduler.close)
        from threading import Event
        stopped = Event()
        stack.callback(stopped.set)
        def on_remote(case):
            with torch.inference_mode():
                return evaluate_case(case, bindings, output, runtime, device, remote,
                                     ensembles, requested)
        def remote_cases(cases):
            for case in cases:
                if stopped.is_set():return
                on_remote(case)
        for scope in scopes:
            from explore_a.evaluation.transport import case_route, assign_cases
            cases = assign_cases(list(cohorts(scope, config, benchmark_root, tum_root, setting, datasets, full_dev)), config)
            for case in cases:
                write_json(output/scope/case["dataset"]/case["key"]/"cohort.json", case)
            owned = [case for case in cases
                     if remote is not None and case_route(case, config) == "dgx"]
            remote_work = scheduler.submit(lambda owned=owned: remote_cases(owned))
            for case in cases:
                if remote is not None and case_route(case, config) == "dgx":continue
                if remote_work.done():remote_work.result()
                with evaluation_state(*[head(b) for b in bindings if "task" in b]):
                    evaluate_case(case, bindings, output, runtime, device, remote,
                                  ensembles, requested)
            remote_work.result()
    return summarize(output)
