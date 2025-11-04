"""Dataset handling for ROI time-series based brain graph construction.

With the synthetic data utilities now in place, this dataset can operate purely
on generated ROI time-series to perform an end-to-end smoke test without any
real fMRI files. The original placeholders for real data loading remain so the
interface stays consistent when real datasets are added later.
"""
from __future__ import annotations

import os
from typing import Dict, List, Any, Optional

import numpy as np
from src.preprocess.compute_edge_features import compute_edge_features
from src.preprocess.build_graph import build_graph_from_edge_features
from src.preprocess.build_hypergraph import build_hypergraph
from src.utils.torch_import import Dataset, torch


class ROITimeSeriesDataset(Dataset):
    """Dataset wrapping ROI time-series and labels for brain disorder classification.

    This dataset assumes there are pre-defined split files (e.g., ``train.txt``)
    that list subject IDs. For the MVP, data loading is mocked with random
    tensors; downstream modules and interfaces are ready for real data.
    """

    def __init__(
        self,
        data_dir: str,
        split_file: Optional[str],
        config: Dict[str, Any],
        *,
        use_synthetic: bool = False,
        synthetic_dir: Optional[str] = None,
        subject_ids: Optional[List[str]] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split_file = split_file
        self.config = config
        self.use_synthetic = use_synthetic
        self.synthetic_dir = synthetic_dir

        if subject_ids is not None:
            self.subject_ids = subject_ids
        elif split_file is not None:
            self.subject_ids = self._load_split_ids(split_file)
        else:
            raise ValueError(
                "Either provide subject_ids directly or specify a split_file."
            )

        if self.use_synthetic and not self.synthetic_dir:
            raise ValueError("synthetic_dir must be provided when use_synthetic=True")

    def _load_split_ids(self, split_file: str) -> List[str]:
        split_path = os.path.join(self.data_dir, "splits", split_file)
        if not os.path.exists(split_path):
            # TODO: Replace with proper error handling when splits are provided.
            raise FileNotFoundError(
                f"Split file {split_path} does not exist. Create it with subject IDs."
            )
        with open(split_path, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
        return ids

    def __len__(self) -> int:
        return len(self.subject_ids)

    def _load_timeseries(self, subject_id: str) -> np.ndarray:
        """Load ROI time-series for a subject.

        TODO: Implement actual loading from preprocessed files. Current placeholder
        returns random data with deterministic seed for reproducibility.
        """

        if self.use_synthetic:
            timeseries_path = os.path.join(
                self.synthetic_dir, f"{subject_id}_timeseries.pt"
            )
            if not os.path.exists(timeseries_path):
                raise FileNotFoundError(
                    f"Synthetic timeseries not found at {timeseries_path}."
                )
            ts_tensor = torch.load(timeseries_path)
            ts_np = ts_tensor.detach().cpu().numpy().astype(np.float32)
            return ts_np

        rng = np.random.default_rng(abs(hash(subject_id)) % (2**32))
        t = self.config.get("time_steps", 200)
        n_rois = self.config.get("num_rois", 116)
        ts = rng.normal(size=(t, n_rois)).astype(np.float32)
        return ts

    def _load_label(self, subject_id: str) -> int:
        """Load the binary diagnosis label.

        TODO: Replace with real label lookup. We currently derive a pseudo label
        from the hash of ``subject_id`` for deterministic behavior.
        """

        if self.use_synthetic:
            label_path = os.path.join(self.synthetic_dir, f"{subject_id}_label.pt")
            if not os.path.exists(label_path):
                raise FileNotFoundError(
                    f"Synthetic label not found at {label_path}."
                )
            label_tensor = torch.load(label_path)
            return int(label_tensor.item())

        return abs(hash(subject_id)) % 2

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        subject_id = self.subject_ids[idx]
        ts = self._load_timeseries(subject_id)
        label = self._load_label(subject_id)

        edge_feat_matrix = compute_edge_features(ts, self.config)
        node_x, edge_index, edge_attr = build_graph_from_edge_features(
            edge_feat_matrix, self.config
        )
        H = build_hypergraph(edge_feat_matrix, self.config)

        sample = {
            "x": node_x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "H": H,
            "y": torch.tensor(label, dtype=torch.long),
            "sid": str(subject_id),
        }
        return sample


if __name__ == "__main__":
    from src.config import get_default_config

    cfg = get_default_config()
    try:
        dataset = ROITimeSeriesDataset(cfg["data_dir"], cfg["train_split"], cfg)
        print("Dataset length:", len(dataset))
        first = dataset[0]
        print(
            "Sample keys:", first.keys(),
            "x shape:", first["x"].shape,
            "edge_index shape:", first["edge_index"].shape,
            "H shape:", first["H"].shape,
        )
    except FileNotFoundError as exc:
        print("Dataset self-test skipped:", exc)
