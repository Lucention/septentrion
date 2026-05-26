"""Deterministic seeding across stdlib, NumPy, and (optionally) PyTorch."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    """Seed all RNGs and return a NumPy Generator to thread through stages."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return np.random.default_rng(seed)
