"""Construct hypergraph incidence matrix from pairwise features."""
from __future__ import annotations

from typing import Dict, Any

from src.utils.torch_import import torch


def build_hypergraph(edge_feat_matrix: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    """Build an incidence matrix ``H`` for a simple hypergraph construction.

    The MVP strategy creates one hyperedge per ROI consisting of the ROI itself
    and its top-k most strongly correlated neighbors (based on the Pearson
    channel of the edge feature matrix).
    """

    N = edge_feat_matrix.shape[0]
    device = edge_feat_matrix.device
    k = min(config.get("k_top_hyper", config.get("k_top_edges", 8)), N - 1)

    pearson = edge_feat_matrix[:, :, 0]
    _, topk_indices = torch.topk(pearson, k=k, dim=-1)

    hyperedges = []
    for i in range(N):
        nodes = set([i])
        nodes.update(topk_indices[i].tolist())
        hyperedges.append(sorted(nodes))

    M = len(hyperedges)
    H = torch.zeros(N, M, dtype=torch.float32, device=device)
    for e_idx, nodes in enumerate(hyperedges):
        for n in nodes:
            H[n, e_idx] = 1.0

    return H


if __name__ == "__main__":
    feats = torch.rand(5, 5, 4)
    H = build_hypergraph(feats, {"k_top_edges": 2})
    print("Hypergraph shape:", H.shape)
