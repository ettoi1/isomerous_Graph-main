"""Random seed utilities supporting reproducible synthetic experiments."""
from __future__ import annotations

import os
import random

try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when numpy is absent
    np = None  # type: ignore
    _NP_AVAILABLE = False

from src.utils.torch_import import TORCH_AVAILABLE, torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    if _NP_AVAILABLE:
        np.random.seed(seed)  # type: ignore[attr-defined]
    torch.manual_seed(seed)
    if TORCH_AVAILABLE:
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    set_seed(42)
    print("Seeds set to 42")
