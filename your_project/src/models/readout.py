"""Graph readout and classification head.

Synthetic样本可以检验读出层是否在伪标签上学习到大于随机的性能。"""
from __future__ import annotations

from src.utils.torch_import import nn, torch


class GraphReadout(nn.Module):
    """Average pooling followed by MLP for graph-level prediction."""

    def __init__(self, d_in: int, num_classes: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_in // 2),
            nn.ReLU(),
            nn.Linear(d_in // 2, num_classes),
        )

    def forward(self, node_x: torch.Tensor) -> torch.Tensor:
        graph_feat = node_x.mean(dim=0, keepdim=True)
        logits = self.mlp(graph_feat)
        return logits


if __name__ == "__main__":
    readout = GraphReadout(16, 2)
    node_x = torch.randn(5, 16)
    logits = readout(node_x)
    print("Logits shape:", logits.shape)
