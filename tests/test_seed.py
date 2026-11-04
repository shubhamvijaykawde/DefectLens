import random

import numpy as np
import torch

from defectlens.seed import seed_everything


def test_same_seed_same_numbers() -> None:
    seed_everything(123)
    a = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    seed_everything(123)
    b = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert a == b


def test_different_seed_differs() -> None:
    seed_everything(1)
    a = float(torch.rand(1))
    seed_everything(2)
    assert a != float(torch.rand(1))


def test_returns_seed() -> None:
    assert seed_everything(7) == 7
