"""End-to-end Brain Graph MoE model composition.

Thanks to the synthetic data pipeline, this module can now be stress-tested
without真实fMRI输入，确保社区、超图与MoE的交互逻辑在端到端训练中运作正常。
"""
from __future__ import annotations

from typing import Dict, Any, Tuple

from src.utils.torch_import import nn, torch

from src.models.community_module import CommunityModule
from src.models.hypergraph_module import HypergraphEncoder
from src.models.moe_layer import RelationMoE
from src.models.readout import GraphReadout


class BrainGraphMoEModel(nn.Module):
    """Compose community, hypergraph, and MoE modules for classification."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        d_node_init = config["d_node_init"]
        d_node_hidden = config["d_node_hidden"]
        d_edge = config["d_edge"]
        d_hyper = config["d_hyper"]
        num_classes = config["num_classes"]

        self.node_proj = nn.Linear(d_node_init, d_node_hidden)
        self.hyper_encoder = HypergraphEncoder(d_node_hidden, d_hyper)
        self.community_module = CommunityModule(d_node_hidden, config["num_communities"])
        self.moe_layer = RelationMoE(
            d_node_hidden, d_edge, config["num_experts"], msg_dim=d_node_hidden
        )
        self.readout = GraphReadout(d_node_hidden, num_classes)

    def forward(
        self,
        node_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        H: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        node_hidden = self.node_proj(node_x)
        node_hyper = self.hyper_encoder(node_hidden, H)
        C = self.community_module(node_hidden)
        edge_comm_ctx = CommunityModule.edge_context(C, edge_index)
        node_updated, moe_aux = self.moe_layer(
            node_hidden, node_hyper, edge_index, edge_attr, edge_comm_ctx
        )
        logits = self.readout(node_updated)
        aux = {"C": C, "moe_aux": moe_aux}
        return logits, aux


if __name__ == "__main__":
    cfg = {
        "d_node_init": 16,
        "d_node_hidden": 32,
        "d_edge": 4,
        "d_hyper": 16,
        "num_classes": 2,
        "num_communities": 4,
        "num_experts": 4,
    }
    model = BrainGraphMoEModel(cfg)
    node_x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    edge_attr = torch.randn(4, 4)
    H = torch.randint(0, 2, (10, 5)).float()
    logits, aux = model(node_x, edge_index, edge_attr, H)
    print("Logits shape:", logits.shape)
    print("Aux keys:", aux.keys())
