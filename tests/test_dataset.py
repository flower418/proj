from pathlib import Path

import numpy as np

from src.data.dataset import inspect_dataset


def test_inspect_dataset_reads_generated_layout(tmp_path: Path):
    np.savez(
        tmp_path / "dataset_full.npz",
        features=np.zeros((12, 8), dtype=np.float32),
        ds_expert=np.linspace(0.01, 0.12, 12, dtype=np.float32),
    )
    np.savez(
        tmp_path / "dataset_full_splits.npz",
        train_indices=np.arange(0, 8),
        val_indices=np.arange(8, 10),
        test_indices=np.arange(10, 12),
    )

    summary = inspect_dataset(tmp_path)

    assert summary["num_samples"] == 12
    assert summary["feature_dim"] == 8
    assert summary["split_sizes"] == {"train": 8, "val": 2, "test": 2}
    assert np.isclose(summary["step_size_min"], 0.01)
    assert np.isclose(summary["step_size_max"], 0.12)
