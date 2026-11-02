"""Frozen backend constructors and CUDA synchronization, without a second runtime."""
import torch

from explore_a.models.flow import PairFlow
from explore_a.models.mftiq import MFTIQVisibility
from explore_a.models.weights import check_weights


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
