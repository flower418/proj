from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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
        splits = np.load(data_path / "dataset_splits.npz")

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
    train = PseudospectrumDataset(data_dir, "train")
    val = PseudospectrumDataset(data_dir, "val")
    test = PseudospectrumDataset(data_dir, "test")

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
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

    # 加载划分
    splits = np.load(data_path / "dataset_splits.npz")
    print(f"\nTrain: {len(splits['train_indices']):,}")
    print(f"Val: {len(splits['val_indices']):,}")
    print(f"Test: {len(splits['test_indices']):,}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    args = parser.parse_args()
    analyze_dataset(args.data_dir)
