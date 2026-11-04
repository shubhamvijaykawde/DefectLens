"""Fix every random source so runs are reproducible."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed python, numpy and (if installed) torch. Returns the seed used.

    ``deterministic=True`` additionally asks torch for deterministic kernels, which can be slower
    and makes some ops raise; leave it off unless exact reproducibility is required.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:  # slim serve env has no torch
        return seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    return seed
