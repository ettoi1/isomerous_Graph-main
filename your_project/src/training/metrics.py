"""Evaluation metrics for Brain Graph MoE.

Synthetic数据集上的ACC可作为快速 sanity check，确认训练闭环可用。"""
from __future__ import annotations

from src.utils.torch_import import torch


def compute_acc(pred: torch.Tensor, y: torch.Tensor) -> float:
    """Compute accuracy for classification."""

    preds = pred.argmax(dim=-1)
    correct = (preds == y).float().mean().item()
    return correct


if __name__ == "__main__":
    pred = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    y = torch.tensor([1, 0])
    print("Accuracy:", compute_acc(pred, y))
