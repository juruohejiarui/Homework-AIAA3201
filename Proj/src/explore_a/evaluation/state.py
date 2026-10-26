"""Temporary evaluation must not alter training modes or any training random stream."""
from contextlib import contextmanager
import random

import numpy as np
import torch


@contextmanager
def evaluation_state(*models):
    modes = {module: module.training for model in models if model is not None
             for module in model.modules()}
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        for model in models:
            if model is not None:
                model.eval()
        with torch.inference_mode():
            yield
    finally:
        # Restore each flag individually, including children intentionally frozen in eval mode.
        for module, training in modes.items():
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
