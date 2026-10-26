"""Local training checkpoints include optimizer, AMP, RNG and the next update index."""
from pathlib import Path
import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, model, optimizer, scheduler, scaler, step, best, identity):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
             "step": step, "best": best, "identity": identity,
             "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
             "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, identity=None):
    # Only our locally produced run checkpoints use pickle RNG state. External weights use
    # weights_only=True in their adapters, never this loader.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if identity is not None and state["identity"] != identity:
        raise ValueError(f"Checkpoint belongs to a different experiment: {path}")
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state
