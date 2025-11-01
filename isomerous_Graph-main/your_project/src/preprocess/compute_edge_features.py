"""Compute pairwise edge features from ROI time-series."""
from __future__ import annotations

from typing import Dict, Any

try:  # Prefer the real dependency when available.
    import torch  # type: ignore
except ImportError:  # pragma: no cover - shim fallback keeps smoke tests alive.
    from src.utils.torch_import import torch  # type: ignore

import numpy as np
from math import log


def compute_edge_features(ts: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    """Compute multi-channel connectivity features between ROI pairs.

    Parameters
    ----------
    ts: torch.Tensor
        ROI time-series with shape ``[T, N]`` where ``T`` is the number of
        timepoints and ``N`` is the number of regions of interest.
    config: Dict[str, Any]
        Configuration dictionary that may specify ``dfc_window`` and
        ``dfc_stride`` hyper-parameters controlling dynamic FC estimation.

    Returns
    -------
    torch.Tensor
        Dense tensor of shape ``[N, N, 4]`` storing Pearson correlation,
        partial correlation, dynamic functional connectivity (dFC) variance, and
        Gaussian-assumption mutual information for each ROI pair.

    Notes
    -----
    * 对角线元素会被显式设置为 0，表示我们忽略自连接。
    * Partial correlation 是通过 precision matrix （协方差矩阵的伪逆）得到，衡量的是排除其它 ROI 影响后的直接耦合强度。
    * dFC 方差通过滑动窗口统计相关矩阵的变化，体现耦合是否随时间波动。
    * 互信息使用高斯近似公式 ``-0.5 * log(1 - r^2)``，与皮尔逊相关 ``r`` 相对应；后续可以替换为更精确的非线性估计器。
    """

    assert ts.ndim == 2, "ts should be [T, N]"
    # Support both numpy arrays and torch tensors
    T, N = ts.shape
    assert T > 5 and N > 1, "need enough timepoints/ROIs"

    if isinstance(ts, torch.Tensor):
        ts_np = ts.detach().cpu().numpy()
    else:
        ts_np = np.asarray(ts, dtype=np.float64)
    ts_np = np.asarray(ts_np, dtype=np.float64)

    T, N = ts_np.shape
    eps = 1e-6

    # ------------------------------------------------------------------
    # 1) Pearson correlation across the full time horizon.
    # ------------------------------------------------------------------
    pearson = np.corrcoef(ts_np, rowvar=False)
    pearson = np.clip(pearson, -1.0 + eps, 1.0 - eps)

    # ------------------------------------------------------------------
    # 2) Partial correlation via precision matrix (covariance pseudo-inverse).
    #    fMRI ROI 信号高度相关时协方差矩阵可能接近奇异，因此使用 pinv
    #    (Moore-Penrose 伪逆) 而不是直接求逆，以提高数值稳定性。
    # ------------------------------------------------------------------
    cov = np.cov(ts_np, rowvar=False, bias=False)
    precision = np.linalg.pinv(cov)
    diag_precision = np.diag(precision)
    diag_precision = np.clip(diag_precision, eps, None)
    denom = np.sqrt(np.outer(diag_precision, diag_precision))
    partial_corr = -precision / denom
    np.fill_diagonal(partial_corr, 1.0)
    partial_corr = np.clip(partial_corr, -1.0 + eps, 1.0 - eps)

    # ------------------------------------------------------------------
    # 3) Dynamic functional connectivity variance.
    #    使用滑动窗口计算相关矩阵，衡量边在时间维度的稳定性。
    # ------------------------------------------------------------------
    win_len = int(config.get("dfc_window", 40))
    stride = int(config.get("dfc_stride", 10))
    corr_windows = []
    if win_len > 0 and stride > 0 and T >= win_len:
        for start in range(0, T - win_len + 1, stride):
            window = ts_np[start : start + win_len]
            corr_win = np.corrcoef(window, rowvar=False)
            corr_windows.append(corr_win)
    if len(corr_windows) >= 2:
        corr_stack = np.stack(corr_windows, axis=0)
        dfc_var = corr_stack.var(axis=0)
    else:
        # 当时间窗数量不足以估计方差时，返回全零矩阵，表示“无法判断波动性”。
        dfc_var = np.zeros((N, N), dtype=np.float64)

    # ------------------------------------------------------------------
    # 4) Gaussian mutual information approximation.
    #    对于零均值高斯变量，互信息可由皮尔逊相关系数直接得到。
    #    这是一种合理的非线性依赖 proxy，后续可以替换为 kNN estimator。
    # ------------------------------------------------------------------
    mi_denom = np.maximum(1e-6, 1.0 - pearson ** 2)
    mutual_info = -0.5 * np.vectorize(log, otypes=[float])(mi_denom)

    # 堆叠所有特征通道并将对角线清零。
    features = np.stack([pearson, partial_corr, dfc_var, mutual_info], axis=-1)
    for c in range(features.shape[-1]):
        np.fill_diagonal(features[:, :, c], 0.0)

    edge_feat_tensor = torch.tensor(features, dtype=torch.float32)
    return edge_feat_tensor


if __name__ == "__main__":
    dummy_ts = torch.randn(120, 10)
    feats = compute_edge_features(dummy_ts, {"dfc_window": 30, "dfc_stride": 10})
    print("Edge feature tensor shape:", feats.shape)
