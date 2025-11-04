from __future__ import annotations

#Entry point for training BrainGraphMoEModel.
#With the synthetic pipeline enabled, ``main_train.py`` can generate mock data on
#the fly, exercise图构建/超图/MoE等模块，并在几分钟内完成端到端的冒烟测试。



# NOTE: Prefer running this module with ``python -m src.main_train`` from the
# repository root so that ``src`` is resolved as a package. The block below
# provides a fallback when executed directly (``python main_train.py``).
if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))


import os
from typing import List

from src.config import get_default_config
from src.utils.seed import set_seed
from src.utils.logging import log
from src.utils.torch_import import TORCH_AVAILABLE, torch


def main() -> None:
    cfg = get_default_config()
    set_seed(cfg["seed"])

    os.makedirs(cfg["output_dir"], exist_ok=True)

    try:
        import numpy as _np  # noqa: F401
        numpy_available = True
    except ImportError:  # pragma: no cover - triggered in minimalist sandboxes
        numpy_available = False

    if not TORCH_AVAILABLE or not numpy_available:
        log(
            "Missing dependencies: torch_available={}, numpy_available={}. "
            "Install the real libraries to run training.",
            TORCH_AVAILABLE,
            numpy_available,
        )
        return

    from src.dataset.roi_dataset import ROITimeSeriesDataset
    from src.dataset.collate import simple_collate_fn
    from src.models.brain_gnn import BrainGraphMoEModel
    from src.training.train_loop import train_epoch, eval_epoch
    from src.utils.synthetic_data import (
        generate_synthetic_dataset,
        save_synthetic_dataset,
    )
    from src.utils.torch_import import DataLoader, optim

    data_dir = cfg["data_dir"]

    if cfg.get("use_synthetic", False):
        synth_dir = os.path.join(data_dir, "raw_synth")
        if not os.path.exists(synth_dir) or not any(
            name.endswith("_timeseries.pt") for name in os.listdir(synth_dir)
        ):
            dataset = generate_synthetic_dataset(
                cfg["synth_num_subjects"],
                cfg["synth_num_timepoints"],
                cfg["synth_num_rois"],
                cfg.params,
            )
            save_synthetic_dataset(dataset, synth_dir)
        else:
            dataset = []
            for fname in sorted(os.listdir(synth_dir)):
                if fname.endswith("_timeseries.pt"):
                    sid = fname.replace("_timeseries.pt", "")
                    dataset.append({"sid": sid})

        subject_ids: List[str]
        if dataset:
            subject_ids = [sample["sid"] for sample in dataset]
        else:
            subject_ids = sorted(
                {
                    name.split("_timeseries.pt")[0]
                    for name in os.listdir(synth_dir)
                    if name.endswith("_timeseries.pt")
                }
            )
        split_idx = max(1, int(len(subject_ids) * 0.8))
        train_ids = subject_ids[:split_idx]
        val_ids = subject_ids[split_idx:]
        if len(val_ids) == 0:
            val_ids = train_ids[-1:]
            train_ids = train_ids[:-1]
        log(
            "Using synthetic data: train subjects={}, val subjects={}",
            len(train_ids),
            len(val_ids),
        )

        train_dataset = ROITimeSeriesDataset(
            data_dir,
            None,
            cfg,
            use_synthetic=True,
            synthetic_dir=synth_dir,
            subject_ids=train_ids,
        )
        val_dataset = ROITimeSeriesDataset(
            data_dir,
            None,
            cfg,
            use_synthetic=True,
            synthetic_dir=synth_dir,
            subject_ids=val_ids,
        )
    else:
        train_dataset = ROITimeSeriesDataset(
            data_dir, cfg["train_split"], cfg, use_synthetic=False
        )
        val_dataset = ROITimeSeriesDataset(
            data_dir, cfg["val_split"], cfg, use_synthetic=False
        )
        log(
            "Using placeholder real data loader: train split={}, val split={}",
            cfg["train_split"],
            cfg["val_split"],
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=simple_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=simple_collate_fn,
    )

    model = BrainGraphMoEModel(cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, cfg, epoch)
        val_acc = eval_epoch(model, val_loader, cfg, epoch, split="val")
        log("Epoch {}: train_loss={:.4f}, val_acc={:.4f}", epoch, train_loss, val_acc)

    # TODO: Implement test evaluation, early stopping, and TensorBoard logging.


if __name__ == "__main__":
    main()
