"""One pair executor contract, in-process or over bounded framed SSH transport."""
from collections import Counter, OrderedDict
from pathlib import Path
import hashlib
import json
import shlex
import subprocess
import threading
import time

import numpy as np
import torch

from explore_a.data.reading import fit_image
from explore_a.online.components import LocalExecutor
from explore_a.remote_online.wire import send, receive


class DirectPipeline:
    """Remote and pure-local execution share exactly the same frozen implementation."""
    def __init__(self, config, device="cuda", capacity=16):
        self.executor, self.device = LocalExecutor(config, device), device
        self.values, self.capacity, self.counts = OrderedDict(), capacity, Counter()

    def get(self, operation, images, canvas, flow=None, native=False):
        key = (operation, flow, tuple(canvas), native, tuple(
            (im.shape, hashlib.sha256(im.tobytes()).hexdigest()) for im in images))
        if key not in self.values:
            self.values[key] = self.executor.execute(operation, images, canvas, flow=flow, native=native)
            self.counts[operation + "_executed"] += 1
        else:
            self.counts[operation + "_reused"] += 1
        self.values.move_to_end(key)
        while len(self.values) > self.capacity:
            self.values.popitem(last=False)
        return self.values[key]

    def prepare(self, row, occurrence, flows=None, phase=None, native=False):
        return {"native": native}

    def depth_pair(self, raw):
        data = {k: v.to(self.device) for k, v in raw["data"].items()}
        for old, prefix in ((True, "previous"), (False, "current")):
            native = raw.get("frozen_ticket", {}).get("native", False)
            value = self.get("prediction", [raw["images"][old]], raw["canvas"], native=native)["prediction"]
            if native:
                raw.setdefault("native_predictions", {})[prefix] = value
                value = fit_image(value, raw["canvas"])[0][None]
            data[prefix + "_depth"] = torch.as_tensor(value, device=self.device)
        return data

    def add_flow(self, raw, base, flow, teacher=False):
        images = [raw["fitted"][True], raw["fitted"][False]]
        values = self.get("flow", images, raw["canvas"], flow)
        data = {**base, **{k: torch.as_tensor(v, device=self.device) for k, v in values.items()}}
        if teacher:
            v = self.get("teacher", [*images, values["backward"]], raw["canvas"], flow)
            data["teacher_visibility"] = torch.as_tensor(v["visibility"], device=self.device)
        return data

    def statistics(self):
        return dict(self.counts)

    def close(self):
        self.values.clear()


class RemotePair:
    """A dedicated DGX scheduler thread owns the connection and all request I/O."""
    def __init__(self, bindings, config):
        from explore_a.online.pool import Workers
        self.pool = Workers(1, "dgx-case")
        self.bindings, self.config, self.process = bindings, config, None
        self.closed, self.state_lock = False, threading.Lock()
        self.counts = Counter()
        self.temporary = []

    def _connect(self):
        if self.closed:
            raise OSError("Remote pair executor is closed")
        if self.process is not None:
            return
        from explore_a.remote_online.policy import settings
        from explore_a.remote_online.sync import ensure_synchronized
        from depth_project.paths import PROJECT
        ensure_synchronized(self.config)
        policy = settings(self.config)
        remote_project = policy["project"] + "/Proj"
        specs = []
        synced = set()
        def sync_file(source):
            source = Path(source).resolve()
            relative = source.relative_to(PROJECT)
            if relative in synced:return
            destination = remote_project + "/" + str(relative.parent)
            subprocess.run(["ssh", "-o", "BatchMode=yes", policy["host"],
                            shlex.join(["mkdir", "-p", destination])], check=True)
            subprocess.run(["rsync", "-a", "--exclude=.git", "--exclude=__pycache__", "--exclude=*.pyc",
                            str(source), policy["host"] + ":" + destination + "/"], check=True)
            synced.add(relative)
        for b in self.bindings:
            e = dict(b["entry"])
            if "weights" in b:
                from depth_project.models import adapter_type
                sync_file(b["weights"])
                e["weights"] = str(Path(b["weights"]).resolve().relative_to(PROJECT))
                for relative in (*getattr(adapter_type(e["adapter"]), "runtime_paths", ()),
                                 *e.get("sync_paths", ())):
                    sync_file(PROJECT/relative)
            if "checkpoint" in b:
                source = b["checkpoint"]
                digest = hashlib.file_digest(source.open("rb"), "sha256").hexdigest()
                relative = "outputs/evaluation_snapshots/" + digest + ".pt"
                destination = remote_project + "/outputs/evaluation_snapshots"
                subprocess.run(["ssh", "-o", "BatchMode=yes", policy["host"],
                                shlex.join(["mkdir", "-p", destination])], check=True)
                subprocess.run(["rsync", "-a", str(source), policy["host"] + ":" +
                                remote_project + "/" + relative], check=True)
                e["checkpoint"] = relative
                e["sha256"] = digest
            c = b["config"]
            cfg = {k: c[k] for k in ("data", "geometry", "training", "frontend", "online")}
            cfg["weights"] = {k: str(Path(v).relative_to(PROJECT)) for k,v in c["weights"].items()}
            cfg["paths"] = {k: str(Path(v).relative_to(PROJECT)) for k,v in c["paths"].items()
                            if k == "dinov2_source"}
            specs.append({"entry": e, "config": cfg})
        command = shlex.join(["env", "PYTHONPATH=" + remote_project + "/src", policy["python"],
                              "-u", "-m", "explore_a.evaluation.worker"])
        connection = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "Compression=no",
                      "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
                      "-o", "ServerAliveCountMax=3", policy["host"], command]
        with self.state_lock:
            if self.closed:
                raise OSError("Remote pair executor is closed")
            self.process = subprocess.Popen(connection, stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE)
        from depth_project.models.provenance import source_signature
        from explore_a.evaluation.session import frozen_identity, weights_hash
        expected = {"source": source_signature(), "weights": frozen_identity(self.bindings),
                    "references": {b["entry"]["name"]: weights_hash(b["weights"])
                                   for b in self.bindings if "weights" in b}}
        send(self.process.stdin, {"type": "hello", "specs": specs, "protocol": "case-pair-v1", "identity": expected})
        reply, _ = receive(self.process.stdout)
        if reply.get("type") != "ready":
            raise RuntimeError(reply)
        if reply.get("identity") != expected:
            raise RuntimeError("Host/DGX inference source or frontend weights differ; synchronize code")
        self.provenance = reply

    def _infer(self, raw, native, history, names, details=False):
        self._connect()
        began = time.monotonic()
        send(self.process.stdin, {"type": "pair", "dataset": raw["row"]["dataset"],
             "sequence": raw["row"]["sequence"], "native": native, "history": history,
             "names": names, "details": details}, [raw["images"][True], raw["images"][False]])
        meta, arrays = receive(self.process.stdout)
        if meta.get("type") != "result":
            raise RuntimeError(meta.get("message", "Remote model inference failed"))
        self.counts["remote_pairs"] += 1
        self.counts["remote_wall_s"] += time.monotonic()-began
        return dict(zip(meta["names"], arrays))

    def submit(self, raw, native, history, names, details=False):
        return self.pool.submit(lambda: self._infer(raw, native, history, names, details))

    def _reference(self, binding, request):
        from tempfile import TemporaryDirectory
        from depth_project.models.api import ModelOutput
        from depth_project.models.serialization import pack, unpack
        self._connect()
        arrays = []
        parameters = pack({"specific_in": request.specific_in,
                           "frame_ids": request.frame_ids, "timestamps": request.timestamps,
                           "reset_history": request.reset_history,
                           "specific_param": binding["entry"].get("specific_param", {})}, arrays)
        frames = request.rgb
        send(self.process.stdin, {"type": "reference_begin", "name": binding["entry"]["name"],
                                  "frames": len(frames), "parameters": parameters}, arrays)
        reply, _ = receive(self.process.stdout)
        if reply.get("type") != "accepted":
            raise RuntimeError(reply)
        for frame in frames:
            send(self.process.stdin, {"type": "reference_frame"}, [np.asarray(frame)])
        meta, arrays = receive(self.process.stdout)
        if meta.get("type") != "reference_result":
            raise RuntimeError(meta)
        extras = unpack(meta["specific_out"], arrays)
        temp = TemporaryDirectory(prefix="remote-prediction-")
        self.temporary.append(temp)
        mapped = np.lib.format.open_memmap(Path(temp.name)/"prediction.npy", mode="w+", dtype=np.float32,
                                           shape=tuple(meta["shape"]))
        for i in range(len(frames)):
            reply, values = receive(self.process.stdout)
            if reply.get("type") != "prediction_frame" or reply["index"] != i:
                raise RuntimeError("Remote reference frame order differs")
            mapped[i] = values[0]
        return ModelOutput(torch.from_numpy(mapped), meta["representation"], request.frame_ids, extras)

    def reference(self, binding, request):
        return self.pool.submit(lambda: self._reference(binding, request)).result()

    def release_references(self):
        """Call after scoring a returned prediction, before requesting another reference."""
        for temp in self.temporary:
            temp.cleanup()
        self.temporary.clear()

    def close(self):
        with self.state_lock:
            self.closed = True
            process = self.process
        self.pool.close()
        # End this owned SSH connection first: its worker may be blocked in receive().
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
            except ProcessLookupError:
                pass
        self.release_references()


def case_route(case, config):
    """The case owner is independent of model list/order and decoder completion."""
    from explore_a.online.routing import profiles
    if (config.get("execution", {}).get("backend") != "hybrid"
            or not config.get("remote_online", {}).get("enabled", False)):
        return "host"
    if "owner" in case:
        return case["owner"]
    p = config.get("evaluation_execution", {}).get("remote_cases", 0.)
    if not 0 <= p <= 1:
        raise ValueError("remote_cases must be in [0,1]")
    token = f"case-v1:{config['data']['seed']}:{case['scope']}:{case['key']}".encode()
    fraction = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / 2**64
    return "dgx" if fraction < p else "host"


def assign_cases(cases, config):
    """Balance known case sizes per scope; never exceed the configured DGX frame budget."""
    from collections import defaultdict
    groups = defaultdict(list)
    for case in cases:
        groups[case["scope"]].append(case)
    probability = config.get("evaluation_execution", {}).get("remote_cases", 0.)
    if not 0 <= probability <= 1:
        raise ValueError("remote_cases must be in [0,1]")
    if (config.get("execution", {}).get("backend") != "hybrid"
            or not config.get("remote_online", {}).get("enabled", False)):
        probability = 0.
    def size(case):
        return len(case.get("rows", case.get("frames", ())))
    for group in groups.values():
        target = probability*sum(map(size, group))
        assigned = 0
        def order(case):
            return hashlib.sha256(f"{config['data']['seed']}:{case['key']}".encode()).digest()
        for case in sorted(group, key=order):
            remote = assigned+size(case) <= target+1e-9
            case["owner"] = "dgx" if remote else "host"
            if remote:
                assigned += size(case)
    return cases
