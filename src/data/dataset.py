from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal environments
    torch = None
    DataLoader = None

    class Dataset:  # type: ignore[override]
        pass


def _resolve_dataset_base(data_dir: str | Path, dataset_name: str | None = None) -> tuple[Path, Path]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")

    if dataset_name is not None:
        data_path = root / f"{dataset_name}.npz"
        split_path = root / f"{dataset_name}_splits.npz"
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        return data_path, split_path

    preferred = root / "dataset_full.npz"
    preferred_splits = root / "dataset_full_splits.npz"
    if preferred.exists() and preferred_splits.exists():
        return preferred, preferred_splits

    candidates = sorted(
        path
        for path in root.glob("*.npz")
        if not path.name.endswith("_splits.npz")
    )
    if not candidates:
        raise FileNotFoundError(f"No dataset .npz files found under {root}")

    for candidate in reversed(candidates):
        split_path = candidate.with_name(f"{candidate.stem}_splits.npz")
        if split_path.exists():
            return candidate, split_path

    raise FileNotFoundError(
        f"Found dataset files under {root}, but none has a matching *_splits.npz file."
    )


class PseudospectrumDataset(Dataset):
    def __init__(self, data_dir: str | Path, split: str = "train", dataset_name: str | None = None):
        if split not in {"train", "val", "test", "all"}:
            raise ValueError("split must be one of: train, val, test, all")

        data_path, split_path = _resolve_dataset_base(data_dir, dataset_name=dataset_name)
        data = np.load(data_path)
        features = np.asarray(data["features"], dtype=np.float32)
        ds_expert = np.asarray(data["ds_expert"], dtype=np.float32)

        if len(features) != len(ds_expert):
            raise ValueError("Dataset arrays features and ds_expert must have the same length.")

        if split == "all":
            indices = np.arange(len(features), dtype=np.int64)
        else:
            splits = np.load(split_path)
            split_key = f"{split}_indices"
            if split_key not in splits:
                raise KeyError(f"Split file {split_path} does not contain {split_key}.")
            indices = np.asarray(splits[split_key], dtype=np.int64)

        self.data_path = data_path
        self.split_path = split_path
        self.split = split
        self.indices = indices
        self.features = features[indices]
        self.ds_expert = ds_expert[indices]

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int):
        if torch is None:
            raise RuntimeError("torch is required to materialize dataset samples.")
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "ds_expert": torch.tensor(self.ds_expert[idx], dtype=torch.float32),
        }


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int = 0,
    dataset_name: str | None = None,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if torch is None or DataLoader is None:
        raise RuntimeError("torch is required to create dataloaders.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if pin_memory is None:
        pin_memory = bool(torch.cuda.is_available())

    train_dataset = PseudospectrumDataset(data_dir, split="train", dataset_name=dataset_name)
    val_dataset = PseudospectrumDataset(data_dir, split="val", dataset_name=dataset_name)
    test_dataset = PseudospectrumDataset(data_dir, split="test", dataset_name=dataset_name)

    loader_kwargs = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": pin_memory,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def inspect_dataset(data_dir: str | Path, dataset_name: str | None = None) -> dict:
    data_path, split_path = _resolve_dataset_base(data_dir, dataset_name=dataset_name)
    data = np.load(data_path)
    splits = np.load(split_path)

    features = np.asarray(data["features"], dtype=np.float32)
    ds_expert = np.asarray(data["ds_expert"], dtype=np.float32)

    summary = {
        "data_file": str(data_path),
        "split_file": str(split_path),
        "num_samples": int(len(features)),
        "feature_dim": int(features.shape[1]) if features.ndim == 2 else None,
        "step_size_min": float(np.min(ds_expert)) if len(ds_expert) > 0 else 0.0,
        "step_size_max": float(np.max(ds_expert)) if len(ds_expert) > 0 else 0.0,
        "split_sizes": {
            "train": int(len(splits["train_indices"])),
            "val": int(len(splits["val_indices"])),
            "test": int(len(splits["test_indices"])),
        },
    }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect a generated pseudospectrum dataset.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset-name", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = inspect_dataset(args.data_dir, dataset_name=args.dataset_name)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
