"""Load upstream evaluator functions without copying their definitions."""
import importlib.util
import sys

from depth_project.paths import THIRD_PARTY


def load_evaluator(tae=False, device="cuda"):
    directory = THIRD_PARTY / "Video-Depth-Anything/benchmark/eval"
    sys.path.insert(0, str(directory))
    path = directory / ("eval_tae.py" if tae else "eval.py")
    spec = importlib.util.spec_from_file_location("vda_evaluator", path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    evaluator.device = device
    return evaluator
