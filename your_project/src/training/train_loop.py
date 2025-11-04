"""Training and evaluation loops.

The loops now support saving auxiliary tensors for synthetic smoke tests, making
it easy to validate community assignments和MoE路由是否合理，即使没有真实数据。
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict
from src.training.losses import MainTaskLoss, community_regularizer, moe_balance_loss
from src.training.metrics import compute_acc
from src.utils.torch_import import DataLoader, TORCH_AVAILABLE, torch

if TYPE_CHECKING:  # pragma: no cover - type checker only import
    from torch.utils.data import DataLoader as TorchDataLoader
else:
    TorchDataLoader = DataLoader

def train_epoch(
    model,
    dataloader: TorchDataLoader,
    optimizer,
    config: Dict[str, Any],
    epoch: int,
) -> float:
    model.train()
    main_loss_fn = MainTaskLoss(config)
    total_loss = 0.0
    # epoch 参数暂未直接用于训练逻辑，保留给未来的调度/日志扩展。
    for batch in dataloader:
        optimizer.zero_grad()
        logits, aux = model(batch["x"], batch["edge_index"], batch["edge_attr"], batch["H"])
        y = batch["y"].unsqueeze(0)
        loss_main = main_loss_fn(logits, y)
        loss_comm = community_regularizer(aux["C"], batch["edge_index"], config)
        loss_moe = moe_balance_loss(aux["moe_aux"]["usage"], config)
        loss = loss_main + config.get("lambda_comm", 0.0) * loss_comm + config.get("lambda_moe", 0.0) * loss_moe
        if TORCH_AVAILABLE:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)


def eval_epoch(
    model, dataloader: TorchDataLoader, config: Dict[str, Any], epoch: int, split: str
) -> float:
    model.eval()
    model.eval()
    outputs_dir = os.path.join(config.get("output_dir", "./outputs"), "aux_cache")
    os.makedirs(outputs_dir, exist_ok=True)
    all_acc = []
    with torch.no_grad():
        for batch in dataloader:
            logits, aux = model(batch["x"], batch["edge_index"], batch["edge_attr"], batch["H"])
            y = batch["y"].unsqueeze(0)
            acc = compute_acc(logits, y)
            all_acc.append(acc)
            sid = str(batch["sid"])
            cache_path = os.path.join(
                outputs_dir, f"debug_epoch{epoch}_{split}_sid{sid}.pt"
            )
            torch.save(
                {
                    "sid": sid,
                    "C": aux["C"],
                    "top_expert": aux["moe_aux"]["top_expert"],
                },
                cache_path,
            )
    return float(sum(all_acc) / max(len(all_acc), 1))


if __name__ == "__main__":
    print("Train loop module ready. Run main_train.py for end-to-end training.")
