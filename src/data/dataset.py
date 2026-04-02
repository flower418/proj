from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def ensure_dataset_splits(
    data_dir: str | Path,
    total_samples: int | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 0,
):
    """Ensure dataset_splits.npz exists; create a default split if missing."""
    data_path = Path(data_dir)
    splits_path = data_path / "dataset_splits.npz"
    legacy_splits_path = data_path / "dataset_full_splits.npz"
    if splits_path.exists():
        return np.load(splits_path)
    if legacy_splits_path.exists():
        with np.load(legacy_splits_path) as legacy_splits:
            train_indices = legacy_splits["train_indices"]
            val_indices = legacy_splits["val_indices"]
            test_indices = legacy_splits["test_indices"]
        np.savez(
            splits_path,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
        )
        return {
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,
        }

    if total_samples is None:
        data = np.load(data_path / "dataset_full.npz")
        total_samples = int(data["features"].shape[0])

    rng = np.random.default_rng(seed)
    indices = np.arange(total_samples, dtype=np.int64)
    rng.shuffle(indices)

    train_end = int(train_ratio * total_samples)
    val_end = train_end + int(val_ratio * total_samples)
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    np.savez(
        splits_path,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )
    return {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }


class PseudospectrumDataset(Dataset):
    """伪谱数据集"""

    def __init__(self, data_dir: str, split: str = "train"):
        """
        :param data_dir: 数据集目录
        :param split: train / val / test
        """
        data_path = Path(data_dir)

        # 加载数据
        data = np.load(data_path / "dataset_full.npz")
        splits = ensure_dataset_splits(data_path, total_samples=int(data["features"].shape[0]))

        self.features = data["features"]
        self.ds_expert = data["ds_expert"]
        self.y_restart = data["y_restart"]

        # 获取索引
        if split == "train":
            self.indices = splits["train_indices"]
        elif split == "val":
            self.indices = splits["val_indices"]
        else:
            self.indices = splits["test_indices"]

        print(f"{split}: {len(self.indices):,} samples")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = self.indices[idx]
        return {
            "features": torch.tensor(self.features[i], dtype=torch.float32),
            "ds_expert": torch.tensor(self.ds_expert[i], dtype=torch.float32),
            "y_restart": torch.tensor(self.y_restart[i], dtype=torch.float32),
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建 train/val/test 数据加载器"""
    pin_memory = torch.cuda.is_available()
    train = PseudospectrumDataset(data_dir, "train")
    val = PseudospectrumDataset(data_dir, "val")
    test = PseudospectrumDataset(data_dir, "test")

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
    )


def analyze_dataset(data_dir: str):
    """分析数据集"""
    data_path = Path(data_dir)

    # 加载统计
    stats_file = data_path / "dataset_stats.json"
    if stats_file.exists():
        import json
        with open(stats_file) as f:
            stats = json.load(f)
        print(f"\n总样本：{stats['total_samples']:,}")
        print(f"重启样本：{stats['restart_samples']:,} ({100*stats['restart_samples']/stats['total_samples']:.2f}%)")

    data = np.load(data_path / "dataset_full.npz")
    total_samples = int(data["features"].shape[0])
    restart_samples = int(np.sum(data["y_restart"]))
    if not stats_file.exists():
        print(f"\n总样本：{total_samples:,}")
        print(f"重启样本：{restart_samples:,} ({100*restart_samples/max(total_samples, 1):.2f}%)")

    # 加载或自动创建划分
    splits = ensure_dataset_splits(data_path, total_samples=total_samples)
    print(f"\nTrain: {len(splits['train_indices']):,}")
    print(f"Val: {len(splits['val_indices']):,}")
    print(f"Test: {len(splits['test_indices']):,}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    args = parser.parse_args()
    analyze_dataset(args.data_dir)
