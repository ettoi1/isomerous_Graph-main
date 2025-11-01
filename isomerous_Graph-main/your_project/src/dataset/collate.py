"""Collate utilities for ROI time-series datasets."""
from __future__ import annotations

from typing import Any, Dict, List


def simple_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the first element of the batch.

    Because the current MVP uses ``batch_size=1``, we simply return the first
    and only sample in the batch. When enabling larger batch sizes, replace this
    with proper graph batching (potentially using PyTorch Geometric utilities).
    """

    if len(batch) != 1:
        # TODO: Implement batching logic for ``batch_size > 1`` scenarios.
        raise ValueError("Current collate only supports batch_size=1.")
    return batch[0]


if __name__ == "__main__":
    dummy = [{"foo": 1}, {"foo": 2}]
    try:
        print(simple_collate_fn([dummy[0]]))
        print("Collate test OK")
    except ValueError as exc:
        print("Expected error:", exc)
