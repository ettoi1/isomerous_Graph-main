"""Soft community assignment module.

Synthetic smoke tests can now validate社区分配在伪数据上的行为，帮助我们快速调试。"""
from __future__ import annotations

from typing import Dict, Any

from src.utils.torch_import import F, nn, torch


class CommunityModule(nn.Module):
    """Learnable community assignment from node features."""

    def __init__(self, d_in: int, num_communities: int):
        super().__init__()
        self.linear = nn.Linear(d_in, num_communities)

    def forward(self, node_x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(node_x)
        C = F.softmax(logits, dim=-1)
        return C

    @staticmethod
    def edge_context(C: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute per-edge community context features.

        Parameters
        ----------
        C: torch.Tensor
            Soft community assignment with shape ``(N, K)``.
        edge_index: torch.Tensor
            Edge indices with shape ``(2, E)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(E, 2)`` containing ``[intra_score, bridge_score]``.
        """

        src, dst = edge_index
        intra = (C[src] * C[dst]).sum(dim=-1, keepdim=True)
        bridge = 1.0 - intra
        return torch.cat([intra, bridge], dim=-1)


if __name__ == "__main__":
    module = CommunityModule(8, 3)
    x = torch.randn(5, 8)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    C = module(x)
    ctx = CommunityModule.edge_context(C, edge_index)
    print("Community assignments shape:", C.shape)
    print("Edge context shape:", ctx.shape)
