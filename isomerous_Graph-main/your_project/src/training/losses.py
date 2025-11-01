"""Loss functions for Brain Graph MoE training."""
from __future__ import annotations

"""Training losses used by the BrainGraphMoE pipeline.

The synthetic dataset enables验证这些loss项的数值是否稳定，为后续真实实验打基础。"""

from typing import Dict, Any

from src.utils.torch_import import nn, torch


class MainTaskLoss(nn.Module):
    """Wrapper around task-specific loss."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        task_type = config.get("task_type", "classification")
        if task_type == "classification":
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"Unsupported task type: {task_type}")

    def forward(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, y)


def community_regularizer(C: torch.Tensor, edge_index: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    """Placeholder for community smoothness regularization."""

    # TODO: Implement community smoothness or entropy-based regularization.
    return torch.tensor(0.0, device=C.device)


def moe_balance_loss(usage: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    """Encourage expert usage to be approximately uniform."""

    num_experts = usage.numel()
    expected = usage.mean()
    # TODO: Explore more principled load balancing strategies (e.g., auxiliary losses from Switch Transformers).
    balance = ((usage - expected) ** 2).mean()
    return balance


if __name__ == "__main__":
    config = {"task_type": "classification"}
    main_loss = MainTaskLoss(config)
    pred = torch.randn(2, 2)
    y = torch.tensor([0, 1])
    print("Main loss:", main_loss(pred, y).item())
    usage = torch.tensor([10.0, 5.0, 8.0, 7.0])
    print("MoE balance loss:", moe_balance_loss(usage, config).item())
