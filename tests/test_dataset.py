from pathlib import Path

import numpy as np

from src.data.dataset import inspect_dataset


def test_inspect_dataset_reads_generated_layout(tmp_path: Path):
    np.savez(
        tmp_path / "dataset_full.npz",
        features=np.zeros((12, 7), dtype=np.float32),
        ds_expert=np.linspace(0.01, 0.12, 12, dtype=np.float32),
        y_restart=np.array([0, 1] * 6, dtype=np.int64),
    )
    np.savez(
        tmp_path / "dataset_full_splits.npz",
        train_indices=np.arange(0, 8),
        val_indices=np.arange(8, 10),
        test_indices=np.arange(10, 12),
    )

    summary = inspect_dataset(tmp_path)

    assert summary["num_samples"] == 12
    assert summary["feature_dim"] == 7
    assert summary["split_sizes"] == {"train": 8, "val": 2, "test": 2}
    assert 0.0 < summary["restart_ratio"] < 1.0
