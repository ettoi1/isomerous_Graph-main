"""Configuration utilities for brain graph MoE project.

Configuration now includes switches for the synthetic data pipeline so the
entire训练流程可以在没有真实fMRI的情况下进行验证和调试。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Config:
    """Dataclass wrapper around plain dictionaries for type hints."""

    params: Dict[str, Any]

    def __getitem__(self, item: str) -> Any:
        return self.params[item]

    def get(self, item: str, default: Any = None) -> Any:
        return self.params.get(item, default)


def get_default_config() -> Config:
    """Return the default configuration dictionary wrapped in :class:`Config`.

    This configuration covers dataset paths, model hyperparameters, and training
    settings for a minimal viable experiment. Future experiments should derive
    from this config and extend/override the values as needed.
    """

    config = {
        "data_dir": "./your_project/data",
        "seed": 42,
        "lr": 1e-3,
        "epochs": 3,
        "batch_size": 1,
        "num_workers": 0,
        "use_synthetic": True,
        "synth_num_subjects": 40,
        "synth_num_timepoints": 200,
        "synth_num_rois": 90,
        "synth_latent_rank": 5,
        "synth_base_scale": 0.6,
        "synth_noise_scale": 0.25,
        "synth_anomaly_scale": 1.2,
        "synth_anomaly_fraction": 0.12,
        "num_rois": 90,
        "time_steps": 200,
        "num_communities": 8,
        "num_experts": 4,
        "d_node_init": 16,
        "d_node_hidden": 64,
        "d_hyper": 32,
        "d_edge": 4,
        "k_top_edges": 8,
        # 滑动窗口超参数，用于动态功能连接 (dFC) 方差计算。
        "dfc_window": 40,
        "dfc_stride": 10,
        "task_type": "classification",
        "num_classes": 2,
        "lambda_comm": 0.0,
        "lambda_moe": 1e-3,
        "train_split": "train.txt",
        "val_split": "val.txt",
        "test_split": "test.txt",
        "output_dir": "./your_project/outputs",
    }
    return Config(config)


if __name__ == "__main__":
    cfg = get_default_config()
    print("Default configuration keys:", list(cfg.params.keys()))