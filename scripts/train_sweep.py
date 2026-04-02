from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

import _bootstrap  # noqa: F401

from src.data.dataset import create_dataloaders
from src.nn.controller import build_controller
from src.nn.loss import ControllerLoss
from src.train.logger import TrainingLogger
from src.train.trainer import ControllerTrainer
from src.utils.config import load_yaml_config


PRESET_SETS: Dict[str, List[Dict[str, Any]]] = {
    "quick": [
        {
            "name": "silu_medium",
            "controller": {"hidden_dims": [128, 128, 64], "dropout": 0.05, "activation": "silu", "head_hidden_dim": 64},
            "training": {"learning_rate": 3.0e-4, "weight_decay": 1.0e-5, "lambda_step": 1.0, "lambda_restart": 3.0, "alpha_restart": 0.75, "focal_gamma": 1.5, "gradient_clip_norm": 1.0},
        },
        {
            "name": "silu_wide",
            "controller": {"hidden_dims": [256, 128, 64], "dropout": 0.05, "activation": "silu", "head_hidden_dim": 64},
            "training": {"learning_rate": 2.0e-4, "weight_decay": 2.0e-5, "lambda_step": 1.0, "lambda_restart": 3.0, "alpha_restart": 0.7, "focal_gamma": 1.5, "gradient_clip_norm": 1.0},
        },
        {
            "name": "step_focus",
            "controller": {"hidden_dims": [128, 128, 64], "dropout": 0.0, "activation": "gelu", "head_hidden_dim": 64, "step_size_min": 1.0e-6, "step_size_max": 0.1},
            "training": {"learning_rate": 2.0e-4, "weight_decay": 1.0e-5, "lambda_step": 1.5, "lambda_restart": 2.5, "alpha_restart": 0.7, "focal_gamma": 1.0, "gradient_clip_norm": 1.0},
        },
        {
            "name": "relu_classifier_heavy",
            "controller": {"hidden_dims": [128, 128, 64], "dropout": 0.05, "activation": "relu", "head_hidden_dim": 32},
            "training": {"learning_rate": 3.0e-4, "weight_decay": 1.0e-5, "lambda_step": 0.75, "lambda_restart": 4.0, "alpha_restart": 0.8, "focal_gamma": 2.0, "gradient_clip_norm": 1.0},
        },
    ],
    "full": [
        {
            "name": "silu_medium",
            "controller": {"hidden_dims": [128, 128, 64], "dropout": 0.05, "activation": "silu", "head_hidden_dim": 64},
            "training": {"learning_rate": 3.0e-4, "weight_decay": 1.0e-5, "lambda_step": 1.0, "lambda_restart": 3.0, "alpha_restart": 0.75, "focal_gamma": 1.5, "gradient_clip_norm": 1.0},
        },
        {
            "name": "silu_wider_head",
            "controller": {"hidden_dims": [192, 128, 64], "dropout": 0.05, "activation": "silu", "head_hidden_dim": 96},
            "training": {"learning_rate": 2.0e-4, "weight_decay": 2.0e-5, "lambda_step": 1.0, "lambda_restart": 3.0, "alpha_restart": 0.7, "focal_gamma": 1.5, "gradient_clip_norm": 1.0},
        },
        {
            "name": "gelu_deep",
            "controller": {"hidden_dims": [128, 128, 128, 64], "dropout": 0.05, "activation": "gelu", "head_hidden_dim": 64},
            "training": {"learning_rate": 2.0e-4, "weight_decay": 1.0e-5, "lambda_step": 1.25, "lambda_restart": 2.5, "alpha_restart": 0.7, "focal_gamma": 1.0, "gradient_clip_norm": 1.0},
        },
        {
            "name": "lowdrop_step_focus",
            "controller": {"hidden_dims": [128, 128, 64], "dropout": 0.0, "activation": "silu", "head_hidden_dim": 64, "step_size_min": 1.0e-6, "step_size_max": 0.1},
            "training": {"learning_rate": 1.5e-4, "weight_decay": 1.0e-5, "lambda_step": 1.5, "lambda_restart": 2.0, "alpha_restart": 0.65, "focal_gamma": 1.0, "gradient_clip_norm": 1.0},
        },
        {
            "name": "batchnorm_wide",
            "controller": {"hidden_dims": [256, 128, 64], "dropout": 0.05, "norm_type": "batchnorm", "activation": "silu", "head_hidden_dim": 64},
            "training": {"learning_rate": 2.0e-4, "weight_decay": 2.0e-5, "lambda_step": 1.0, "lambda_restart": 3.0, "alpha_restart": 0.75, "focal_gamma": 1.5, "gradient_clip_norm": 1.0},
        },
        {
            "name": "classifier_heavy",
            "controller": {"hidden_dims": [128, 128, 64], "dropout": 0.05, "activation": "relu", "head_hidden_dim": 32},
            "training": {"learning_rate": 3.0e-4, "weight_decay": 1.0e-5, "lambda_step": 0.75, "lambda_restart": 4.5, "alpha_restart": 0.8, "focal_gamma": 2.0, "gradient_clip_norm": 1.0},
        },
        {
            "name": "tight_step_cap",
            "controller": {"hidden_dims": [128, 128, 64], "dropout": 0.05, "activation": "silu", "head_hidden_dim": 64, "step_size_min": 1.0e-6, "step_size_max": 0.05},
            "training": {"learning_rate": 2.0e-4, "weight_decay": 1.0e-5, "lambda_step": 1.5, "lambda_restart": 2.5, "alpha_restart": 0.7, "focal_gamma": 1.0, "gradient_clip_norm": 1.0},
        },
    ],
}


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="批量试验多组训练超参数。")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preset-set", choices=sorted(PRESET_SETS.keys()), default="quick")
    parser.add_argument("--experiment-prefix", default="sweep")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trial-limit", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default="models")
    parser.add_argument("--log-dir", default="logs")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    train_cfg = load_yaml_config(args.training_config, validate=False).get("defaults", {})
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA 训练，但当前 PyTorch 未检测到可用 GPU。")

    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", config["training"]["batch_size"])
    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", config["training"]["epochs"])
    early_stop_patience = train_cfg.get("early_stop_patience", config["training"].get("early_stop_patience", 10))

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir,
        batch_size=batch_size,
        num_workers=args.num_workers,
    )

    trials = PRESET_SETS[args.preset_set]
    if args.trial_limit is not None:
        trials = trials[: args.trial_limit]

    summary_dir = Path(args.checkpoint_dir) / args.experiment_prefix
    summary_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for trial_index, trial in enumerate(trials):
        seed = args.seed + trial_index
        set_seed(seed)
        effective = deep_update(config, trial)
        experiment_name = f"{args.experiment_prefix}/{trial['name']}"
        print(f"\n=== Trial {trial_index + 1}/{len(trials)}: {trial['name']} (seed={seed}) ===")

        model = build_controller(effective["controller"], input_dim=7).to(device)
        loss_fn = ControllerLoss(
            lambda_step=effective["training"]["lambda_step"],
            lambda_restart=effective["training"]["lambda_restart"],
            alpha_restart=effective["training"]["alpha_restart"],
            focal_gamma=effective["training"]["focal_gamma"],
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=effective["training"]["learning_rate"],
            weight_decay=effective["training"].get("weight_decay", 0.0),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=train_cfg.get("scheduler_factor", effective["training"]["scheduler_factor"]),
            patience=train_cfg.get("scheduler_patience", effective["training"]["scheduler_patience"]),
        )
        logger = TrainingLogger(log_dir=args.log_dir, experiment_name=experiment_name)
        logger.save_config(
            {
                "base_config": config,
                "trial_overrides": trial,
                "effective_config": effective,
                "args": vars(args),
                "seed": seed,
            }
        )
        trainer = ControllerTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            logger=logger,
            scheduler=scheduler,
            gradient_clip_norm=effective["training"].get("gradient_clip_norm"),
        )
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            early_stop_patience=early_stop_patience,
            checkpoint_dir=str(Path(args.checkpoint_dir) / experiment_name),
        )
        test_metrics = trainer.evaluate(test_loader)
        test_metrics.pop("_raw", None)
        logger.close()

        best_epoch_record = min(history, key=lambda item: item["val"]["loss"])
        result = {
            "trial_name": trial["name"],
            "seed": seed,
            "checkpoint": str(Path(args.checkpoint_dir) / experiment_name / "best_model.pt"),
            "training_summary": str(Path(args.log_dir) / experiment_name / "training_summary.png"),
            "epochs_run": len(history),
            "best_val_loss": float(best_epoch_record["val"]["loss"]),
            "best_val_f1": float(best_epoch_record["val"].get("f1", 0.0)),
            "best_val_step_rmse": float(best_epoch_record["val"].get("step_size_rmse", 0.0)),
            "test_loss": float(test_metrics["loss"]),
            "test_f1": float(test_metrics.get("f1", 0.0)),
            "test_accuracy": float(test_metrics.get("accuracy", 0.0)),
            "test_step_rmse": float(test_metrics.get("step_size_rmse", 0.0)),
            "effective_config": effective,
        }
        results.append(result)
        print(json.dumps(result, indent=2))

    results.sort(key=lambda item: (item["test_loss"], item["test_step_rmse"], -item["test_f1"]))
    summary_path = summary_dir / "sweep_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(f"\nSweep summary saved to: {summary_path}")
    if results:
        print("\nTop trial:")
        print(json.dumps(results[0], indent=2))


if __name__ == "__main__":
    main()
