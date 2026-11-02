"""Run the actual saved fusion head on independent NYU labeled images, with no history."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from depth_project.io import read_rgb, write_json
from explore_a.config import load_config
from explore_a.data.reading import fit_image
from explore_a.evaluation.nyu_labeled import metric_function, score
from explore_a.evaluation.part2_protocol import restore_canvas
from explore_a.training.checkpoint import load_model_state
from explore_a.training.prediction import make_fusion, predict


@torch.inference_mode()
def run(config, root, checkpoint, output, device="cuda"):
    from explore_a.frontend import relative
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "checkpoint.pt"
    if snapshot.exists():
        state = torch.load(snapshot, map_location="cpu", weights_only=False)
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = {key: state[key] for key in ("model", "identity", "step", "best")}
        torch.save(state, snapshot)
    from explore_a.training.identity import check_frontend
    check_frontend(state["identity"], config)
    config["training"].update(state["identity"]["training"])
    task = state["identity"]["experiment"]
    if task["occlusion"] != "rule":
        raise ValueError("Independent labeled-image evaluation currently supports rule heads")
    model = make_fusion(task["fusion"], config)
    load_model_state(model, state["model"])
    model.to(device).eval()
    model.requires_grad_(False)
    canvas = config["data"]["canvas_indoor"]
    ids = json.loads((root / "test/manifest.json").read_text())["test_ids"]
    identity = {"checkpoint_step": state["step"], "task": task, "root": str(root),
                "canvas": canvas,
                "input": "one labeled image; previous depth/support explicitly invalid",
                "alignment": "per-image inverse SSI" if relative(config) else "none",
                "device": device,
                "frontend": config.get("frontend", {"name":"da2"}),
                "execution": "blocking frozen components v2"}
    path = output / "run.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Labeled evaluation directory belongs to another experiment")
    write_json(path, identity)
    records, metric = [], metric_function()
    from explore_a.online.factory import make_runtime
    from explore_a.cache.native import labeled_raw
    runtime = make_runtime(config, ["farneback"], False, device)
    raw_iterator = labeled_raw(config, root)
    def measured(value, truth):
        if relative(config):
            from depth_project.evaluation.metrics import align_inverse_depth
            valid = (truth > .001) & (truth < 10)
            crop = np.zeros_like(valid)
            crop[45:471, 41:601] = True
            value, _ = align_inverse_depth(value, truth, valid & crop)
        return score(value, truth, metric)
    try:
        for index, raw in raw_iterator:
            image_id = ids[index]
            name = f"{image_id:04d}"
            receipt = output / "images" / (name + ".json")
            if receipt.exists():
                records.append(json.loads(receipt.read_text()))
                continue
            from explore_a.online.statistics import snapshot, difference
            input_before = snapshot(runtime)
            rgb = raw["images"][False]
            raw["frozen_ticket"] = runtime.prepare(raw["row"], index, {}, phase="nyu", native=True)
            base = runtime.depth_pair(raw)
            native = raw["native_predictions"]["current"]
            fitted = base["current_depth"][0].cpu().numpy()
            data = {key:value.unsqueeze(0) for key,value in base.items()}
            current = data["current_depth"]
            data["forward"] = torch.zeros((1, 2, *canvas), device=device)
            data["backward"] = torch.zeros((1, 2, *canvas), device=device)
            amp = device == "cuda" and config["training"]["amp"]
            with torch.autocast(device_type=device, enabled=amp):
                predicted, _ = predict(model, data, config, "rule")
            exact = torch.equal(predicted["depth"], current)
            if not exact:
                raise ValueError(f"Saved head violated exact no-history fallback: {name}")
            field = predicted["depth"][0, 0].float().cpu().numpy()
            depth = restore_canvas(field, rgb.shape, canvas)
            control = restore_canvas(fitted, rgb.shape, canvas)
            truth = np.load(root / "test/gt" / (name + ".npy")).astype(np.float64)
            record = {"image_id": image_id, "fallback_exact": exact,
                      "input_stats": difference(runtime, input_before),
                      "da2_native_ssi": measured(native, truth),
                      "da2_canvas_control_ssi": measured(control, truth),
                      "best": measured(depth, truth)}
            write_json(receipt, record)
            records.append(record)
            if len(records) % 100 == 0:
                print("BEST NYU", len(records), "/", len(ids), flush=True)
    finally:
        runtime.close()
    branches = ("da2_native_ssi", "da2_canvas_control_ssi", "best")
    result = {**identity, "count": len(records), "per_image": records,
              "fallback_exact_images": sum(r["fallback_exact"] for r in records),
              "mean": {b: {k: float(np.mean([r[b][k] for r in records]))
                           for k in records[0][b]} for b in branches},
              "protocol": "480x640; crop[45:471,41:601]; valid GT(0.001,10)m; clip; image mean",
              "conclusion": "Best equals canvas control; native/canvas differences are resampling"}
    write_json(output / "metrics.json", result)
    print("BEST NYU COMPLETE", result["mean"], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "root", "checkpoint", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    run(load_config(args.config), args.root, args.checkpoint, args.output,
        args.device)


if __name__ == "__main__":
    main()
