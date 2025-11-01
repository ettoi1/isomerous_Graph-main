"""Hypergraph encoder used within the BrainGraphMoE pipeline.

Synthetic datasets ensure this encoder receives realistic-yet-controlled
incidence matrices, enabling end-to-end regression tests without真实数据。"""

from __future__ import annotations

from typing import Dict, Any

from src.utils.torch_import import nn, torch


class HypergraphEncoder(nn.Module):
    """Encode node features via simple hypergraph message passing."""

    def __init__(self, d_in: int, d_hyper: int):
        super().__init__()
        self.node_proj = nn.Linear(d_in, d_hyper)
        self.hyper_proj = nn.Linear(d_hyper, d_hyper)
        self.node_update = nn.Linear(d_hyper, d_in)

    def forward(self, node_x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Apply hypergraph encoding with mean aggregation.

        Parameters
        ----------
        node_x: torch.Tensor
            Node features ``(N, d_in)``.
        H: torch.Tensor
            Incidence matrix ``(N, M)`` where ``M`` is number of hyperedges.
        """

        node_hidden = self.node_proj(node_x)  # (N, d_hyper)
        hyper_feat = H.t() @ node_hidden  # (M, d_hyper)
        degree = H.sum(dim=0, keepdim=True).clamp(min=1.0)
        hyper_feat = hyper_feat / degree.t()
        hyper_feat = self.hyper_proj(hyper_feat)

        node_agg = H @ hyper_feat  # (N, d_hyper)
        node_agg = node_agg / H.sum(dim=1, keepdim=True).clamp(min=1.0)
        node_resid = self.node_update(node_agg)
        return node_x + node_resid


if __name__ == "__main__":
    encoder = HypergraphEncoder(16, 8)
    node_x = torch.randn(5, 16)
    H = torch.randint(0, 2, (5, 5)).float()
    out = encoder(node_x, H)
    print("Encoded shape:", out.shape)
