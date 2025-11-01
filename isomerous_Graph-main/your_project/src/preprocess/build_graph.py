"""Utilities for constructing sparse graphs from edge feature matrices."""
from __future__ import annotations

from typing import Dict, Any, Tuple

from src.utils.torch_import import torch


def build_graph_from_edge_features(
    edge_feat_matrix: torch.Tensor, config: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct sparse graph tensors from dense edge feature matrix.

    Parameters
    ----------
    edge_feat_matrix: torch.Tensor
        Tensor with shape ``(N, N, d_edge)`` containing connectivity features.
    config: Dict[str, Any]
        Configuration dictionary including ``k_top_edges`` and node feature dims.

    Returns
    -------
    node_x: torch.Tensor
        Node feature tensor with shape ``(N, d_node_init)``.
    edge_index: torch.Tensor
        Edge indices with shape ``(2, E)`` following PyTorch Geometric convention
        (bidirectional edges are explicitly duplicated).
    edge_attr: torch.Tensor
        Edge attributes with shape ``(E, d_edge)``.
    """

    N = edge_feat_matrix.shape[0]
    device = edge_feat_matrix.device
    d_edge = edge_feat_matrix.shape[-1]
    k = min(config.get("k_top_edges", 8), N - 1)

    # 通道0存储的是皮尔逊相关，作为构图时的主排序依据（top-k）。
    pearson = edge_feat_matrix[:, :, 0]
    _, topk_indices = torch.topk(pearson, k=k, dim=-1)

    src_list = []
    dst_list = []
    edge_attrs = []
    for i in range(N):
        for idx in topk_indices[i]:
            j = idx.item()
            if i == j:
                continue
            src_list.append(i)
            dst_list.append(j)
            edge_attrs.append(edge_feat_matrix[i, j])
            # Mirror edge to maintain undirected connectivity via directed edges.
            src_list.append(j)
            dst_list.append(i)
            edge_attrs.append(edge_feat_matrix[j, i])

    if len(src_list) == 0:
        # TODO: Handle degenerate cases with proper fallback graph construction.
        raise RuntimeError("No edges were constructed from edge features.")

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
    edge_attr = torch.stack(edge_attrs, dim=0).to(device)

    # Node initialization: simple statistics of correlations as placeholder.
    node_strength = pearson.abs().sum(dim=-1, keepdim=True)
    node_degree = torch.tensor([k] * N, dtype=torch.float32, device=device).view(N, 1)
    node_feat = torch.cat([node_strength, node_degree], dim=-1)
    d_node_init = config.get("d_node_init", node_feat.shape[-1])
    if node_feat.shape[-1] < d_node_init:
        pad = torch.zeros(N, d_node_init - node_feat.shape[-1], device=device)
        node_feat = torch.cat([node_feat, pad], dim=-1)
    elif node_feat.shape[-1] > d_node_init:
        node_feat = node_feat[:, :d_node_init]

    return node_feat.to(torch.float32), edge_index, edge_attr.to(torch.float32)


if __name__ == "__main__":
    feats = torch.rand(5, 5, 4)
    node_x, edge_index, edge_attr = build_graph_from_edge_features(feats, {"k_top_edges": 2, "d_node_init": 4})
    print("node_x shape", node_x.shape)
    print("edge_index shape", edge_index.shape)
    print("edge_attr shape", edge_attr.shape)