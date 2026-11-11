"""DGX pair worker. No GT, dataset reading or alternate model implementations."""
import os
import sys


def main():
    output = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    from pathlib import Path
    import torch
    from depth_project.paths import PROJECT
    from explore_a.remote_online.wire import send, receive
    from explore_a.evaluation.session import head, forward_frame, file_hash, reference, frozen_identity, weights_hash
    from explore_a.evaluation.transport import DirectPipeline
    from explore_a.evaluation.part2 import raw_input
    hello, _ = receive(sys.stdin.buffer)
    if hello.get("protocol") != "case-pair-v1":
        raise ValueError("Remote case protocol mismatch")
    bindings = []
    for spec in hello["specs"]:
        e, c = spec["entry"], spec["config"]
        c["weights"] = {k: str(PROJECT/v) for k,v in c["weights"].items()}
        c["paths"] = {k: str(PROJECT/v) for k,v in c.get("paths", {}).items()}
        if e.get("kind", "fusion") != "fusion":
            bindings.append({"entry": e, "config": c, "weights": PROJECT/e["weights"],
                             "device": "cuda", "model": None})
            continue
        path = PROJECT/e["checkpoint"]
        if file_hash(path) != e["sha256"]:
            raise ValueError("Remote checkpoint checksum mismatch")
        state = torch.load(path, map_location="cpu", weights_only=False)
        bindings.append({"entry": e, "config": c, "state": state,
                         "task": state["identity"]["experiment"], "device": "cuda", "model": None})
    runtime = DirectPipeline(bindings[0]["config"])
    for b in bindings:
        if "task" in b:
            head(b)
    from depth_project.models.provenance import source_signature
    actual = {"source": source_signature(), "weights": frozen_identity(bindings),
              "references": {b["entry"]["name"]: weights_hash(b["weights"]) for b in bindings if "weights" in b}}
    if hello.get("identity") != actual:
        send(output, {"type": "identity_error", "message": "Source/weight checksum differs"})
        return
    send(output, {"type": "ready", "protocol": "case-pair-v1", "identity": actual, "gpu": torch.cuda.get_device_name(),
                  "torch": torch.__version__, "frontend_sha256": file_hash(bindings[0]["config"]["weights"]["da2"])})
    try:
        while True:
            message, images = receive(sys.stdin.buffer)
            if message["type"] == "reference_begin":
                from tempfile import TemporaryDirectory
                import numpy as np
                from depth_project.models.api import ModelInput
                from depth_project.models.serialization import pack, unpack
                parameters = unpack(message["parameters"], images)
                binding = next(b for b in bindings if b["entry"]["name"] == message["name"])
                send(output, {"type": "accepted"})
                with TemporaryDirectory(prefix="remote-rgb-") as temp:
                    mapped = None
                    for i in range(message["frames"]):
                        chunk, values = receive(sys.stdin.buffer)
                        if chunk["type"] != "reference_frame" or len(values) != 1:
                            raise ValueError("Invalid reference RGB stream")
                        if mapped is None:
                            mapped = np.lib.format.open_memmap(Path(temp)/"rgb.npy", mode="w+", dtype=np.uint8,
                                shape=(message["frames"], *values[0].shape))
                        mapped[i] = values[0]
                    with torch.inference_mode():
                        result = reference(binding).forward(ModelInput(rgb=mapped,
                            frame_ids=tuple(parameters["frame_ids"]), timestamps=tuple(parameters["timestamps"]),
                            reset_history=tuple(parameters["reset_history"]),
                            specific_in=parameters["specific_in"]), **parameters["specific_param"])
                    if result.prediction.ndim != 3 or result.prediction.shape[0] != message["frames"]:
                        raise ValueError("Reference adapters must return one HW prediction per RGB frame")
                    arrays = []
                    extras = pack(result.specific_out, arrays)
                    send(output, {"type": "reference_result", "shape": list(result.prediction.shape),
                                  "representation": result.representation, "specific_out": extras}, arrays)
                    for i, value in enumerate(result.numpy()):
                        send(output, {"type": "prediction_frame", "index": i}, [value])
                    del mapped, result
                continue
            if len(images) != 2:
                raise ValueError("Expected previous/current RGB only")
            active = [b for b in bindings if b["entry"]["name"] in message["names"]]
            if not active:
                active = [{"config": bindings[0]["config"], "entry": {"kind": "frozen", "name": "baseline"}}]
            c = active[0]["config"]
            raw = raw_input(images[0], images[1], message["dataset"], message["sequence"], "0", "1", c)
            raw["frozen_ticket"] = {"native": message["native"]}
            try:
                with torch.inference_mode():
                    values = forward_frame(active, raw, runtime, message["native"], message["history"],
                                           message.get("details", False))
                send(output, {"type": "result", "names": list(values)}, list(values.values()))
            except Exception as error:
                send(output, {"type": "error", "message": str(error)})
    except EOFError:
        pass
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
