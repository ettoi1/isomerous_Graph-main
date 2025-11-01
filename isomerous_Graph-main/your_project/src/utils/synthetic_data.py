"""Synthetic data generation utilities for rapid smoke testing."""
from __future__ import annotations

import os
from typing import Dict, Any, List

from src.utils.logging import log
from src.utils.torch_import import TORCH_AVAILABLE, torch


def generate_synthetic_subject(
    num_timepoints: int,
    num_rois: int,
    label: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate a single synthetic subject with controllable coupling patterns.

    本函数通过低秩潜在动态 + 噪声的方式生成 ROI 时序数据，
    并在 label=1（患者）时向部分 ROI 子集注入更强的同步/不稳定耦合。
    这些差异将影响皮尔逊相关、dFC 方差等特征，使得后续模型能够在
    ``train_loop`` 中学习到高于随机水平 (>50%) 的分类效果。
    """

    device = torch.device("cpu")
    rank = config.get("synth_latent_rank", 5)
    base_scale = config.get("synth_base_scale", 0.6)
    noise_scale = config.get("synth_noise_scale", 0.2)
    anomaly_scale = config.get("synth_anomaly_scale", 1.2)
    anomaly_fraction = config.get("synth_anomaly_fraction", 0.15)

    # 低秩共享动态：模拟群体性的脑区协同，确保基本相关性结构。
    shared_latent = torch.randn(num_timepoints, rank, device=device)
    mixing_matrix = torch.randn(rank, num_rois, device=device)
    base_dynamics = base_scale * shared_latent @ mixing_matrix

    # 基础噪声，保证每个 ROI 都有独立成分，避免过拟合。
    noise = noise_scale * torch.randn(num_timepoints, num_rois, device=device)

    timeseries = base_dynamics + noise

    # 选取异常 ROI 子集，模拟病理性耦合。
    num_anomaly = max(1, int(num_rois * anomaly_fraction))
    anomaly_indices = torch.arange(num_anomaly, device=device)

    anomaly_pattern = torch.sin(
        torch.linspace(0, 6.28, num_timepoints, device=device)
    ).unsqueeze(1)
    anomaly_pattern += 0.5 * torch.randn_like(anomaly_pattern)

    if label == 1:
        # 患者组：更强、更不稳定的共同动态，带来更高相关和更大方差。
        jitter = torch.randn(num_timepoints, num_anomaly, device=device)
        pathological_component = anomaly_scale * anomaly_pattern + 0.3 * jitter
    else:
        # 对照组：保持平稳的小幅协同。
        pathological_component = 0.3 * anomaly_pattern

    timeseries[:, anomaly_indices] += pathological_component

    # TODO: 可进一步引入 ROI 级别的漂移或结构性噪声，以模拟更复杂的病理。

    return {
        "timeseries": timeseries.to(torch.float32),
        "label": int(label),
    }


def generate_synthetic_dataset(
    num_subjects: int,
    num_timepoints: int,
    num_rois: int,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Generate a balanced synthetic cohort for quick experiments."""

    dataset: List[Dict[str, Any]] = []
    half = max(1, num_subjects // 2)
    for idx in range(num_subjects):
        label = 0 if idx < half else 1
        subject = generate_synthetic_subject(num_timepoints, num_rois, label, config)
        subject["sid"] = f"synthetic_{idx:04d}"
        dataset.append(subject)
    return dataset


def save_synthetic_dataset(dataset_list: List[Dict[str, Any]], out_dir: str) -> None:
    """Persist synthetic dataset to disk for reuse with the dataset loader."""

    os.makedirs(out_dir, exist_ok=True)
    for sample in dataset_list:
        sid = sample["sid"]
        ts_path = os.path.join(out_dir, f"{sid}_timeseries.pt")
        label_path = os.path.join(out_dir, f"{sid}_label.pt")
        torch.save(sample["timeseries"], ts_path)
        torch.save(torch.tensor(sample["label"], dtype=torch.long), label_path)

    log("Saved {} synthetic subjects to {}", len(dataset_list), out_dir)
    if not TORCH_AVAILABLE:
        log("Synthetic tensors rely on numpy-backed torch stub; gradients are disabled.")


if __name__ == "__main__":
    demo_config: Dict[str, Any] = {
        "synth_latent_rank": 6,
        "synth_base_scale": 0.5,
        "synth_noise_scale": 0.3,
        "synth_anomaly_scale": 1.5,
        "synth_anomaly_fraction": 0.1,
    }
    dataset = generate_synthetic_dataset(20, 200, 90, demo_config)
    out_directory = "./your_project/data/raw_synth"
    save_synthetic_dataset(dataset, out_directory)
    stacked = torch.stack([sample["timeseries"].mean(dim=0) for sample in dataset])
    print("Synthetic dataset saved to", out_directory)
    print("ROI mean mean:", stacked.mean().item())
    print("ROI mean std:", stacked.std().item())
