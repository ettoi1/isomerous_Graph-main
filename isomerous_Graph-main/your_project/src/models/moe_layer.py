"""Mixture-of-experts layer operating on graph relations.

Synthetic数据驱动的测试帮助我们观察专家负载和路由是否符合预期，
从而在真实实验前先完成调试。
"""

from __future__ import annotations

from typing import Dict, Any, Tuple

from src.utils.torch_import import F, nn, torch


class EdgeExpert(nn.Module):
    """Small MLP expert that transforms edge contextual features into messages."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.ReLU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class RelationMoE(nn.Module):
    """Relation-level mixture-of-experts with top-1 routing."""

    def __init__(self, d_in: int, d_edge: int, num_experts: int, msg_dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.msg_dim = msg_dim
        gate_input_dim = d_in * 4 + d_edge + 2
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, gate_input_dim // 2),
            nn.ReLU(),
            nn.Linear(gate_input_dim // 2, num_experts),
        )
        self.experts = nn.ModuleList(
            [EdgeExpert(gate_input_dim, msg_dim) for _ in range(num_experts)]
        )
        self.node_update = nn.Sequential(
            nn.Linear(d_in + msg_dim, d_in),
            nn.ReLU(),
            nn.Linear(d_in, d_in),
        )

    def forward(
        self,
        node_x: torch.Tensor,
        node_hyper: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_comm_ctx: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        src, dst = edge_index
        h_src = node_x[src]
        h_dst = node_x[dst]
        h_src_hyper = node_hyper[src]
        h_dst_hyper = node_hyper[dst]
        gate_input = torch.cat(
            [h_src, h_dst, h_src_hyper, h_dst_hyper, edge_attr, edge_comm_ctx], dim=-1
        )
        logits = self.gate_mlp(gate_input)
        top_expert = torch.argmax(logits, dim=-1)

        msgs = torch.zeros(gate_input.size(0), self.msg_dim, device=node_x.device)
        for expert_id, expert in enumerate(self.experts):
            mask = top_expert == expert_id
            if mask.any():
                msgs[mask] = expert(gate_input[mask])

        node_updates = torch.zeros_like(node_x)
        node_updates = node_updates.index_add(0, dst, msgs)
        node_updated = self.node_update(torch.cat([node_x, node_updates], dim=-1))
        node_updated = node_x + node_updated

        usage = torch.bincount(top_expert, minlength=self.num_experts).to(torch.float32)
        aux = {"top_expert": top_expert, "usage": usage}
        return node_updated, aux


if __name__ == "__main__":
    layer = RelationMoE(16, 4, 4, 8)
    node_x = torch.randn(5, 16)
    node_hyper = torch.randn(5, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    edge_attr = torch.randn(4, 4)
    edge_comm_ctx = torch.randn(4, 2)
    out, aux = layer(node_x, node_hyper, edge_index, edge_attr, edge_comm_ctx)
    print("Updated nodes shape:", out.shape)
    print("Usage:", aux["usage"])
