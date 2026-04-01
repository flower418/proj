from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401

from src.core.pseudoinverse import PseudoinverseSolver
from src.nn.controller import NNController
from src.nn.loss import ControllerLoss
from src.train.data_generator import ExpertDataGenerator, ExpertDataset
from src.train.logger import TrainingLogger
from src.train.trainer import ControllerTrainer
from src.utils.config import load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train the neural controller.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--matrix-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--z0-real", type=float, default=0.2)
    parser.add_argument("--z0-imag", type=float, default=0.1)
    parser.add_argument("--checkpoint-dir", default="models")
    parser.add_argument("--data-out", default="data/train_records.npz")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    train_cfg = load_yaml_config(args.training_config, validate=False).get("defaults", {})

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    A = np.random.randn(args.matrix_size, args.matrix_size) + 1j * np.random.randn(args.matrix_size, args.matrix_size)
    z0 = complex(args.z0_real, args.z0_imag)
    solver = PseudoinverseSolver(
        method=config["solver"]["method"],
        tol=config["solver"]["tol"],
        max_iter=config["solver"]["max_iter"],
    )
    generator = ExpertDataGenerator(
        A=A,
        epsilon=config["ode"]["epsilon"],
        expert_tol=1e-8,
        atol=1e-8,
        drift_threshold=config["tracker"]["restart_drift_threshold"],
        base_step_size=config["ode"]["initial_step_size"],
        max_step_size=config["ode"]["max_step_size"],
        closure_tol=config["tracker"]["closure_tol"],
        solver=solver,
    )
    records = generator.generate_trajectory(z0=z0, max_steps=args.max_steps)
    records.extend(generator.add_state_perturbations(records, noise_std=config["training"]["noise_std"]))
    dataset = ExpertDataset(records)
    split = max(1, int(0.8 * len(dataset)))
    train_dataset = torch.utils.data.Subset(dataset, range(split))
    val_dataset = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    model = NNController(
        input_dim=7,
        hidden_dims=config["controller"]["hidden_dims"],
        dropout=config["controller"]["dropout"],
        norm_type=config["controller"]["norm_type"],
        step_size_min=config["controller"]["step_size_min"],
        step_size_max=config["controller"]["step_size_max"],
    )
    loss_fn = ControllerLoss(
        lambda_step=config["training"]["lambda_step"],
        lambda_restart=config["training"]["lambda_restart"],
        alpha_restart=config["training"]["alpha_restart"],
        focal_gamma=config["training"]["focal_gamma"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get("learning_rate", config["training"]["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_cfg.get("scheduler_factor", config["training"]["scheduler_factor"]),
        patience=train_cfg.get("scheduler_patience", config["training"]["scheduler_patience"]),
    )
    logger = TrainingLogger(log_dir="logs", experiment_name=args.experiment_name)
    logger.save_config({"default": config, "training": train_cfg, "args": vars(args)})
    logger.log_feature_distribution(np.stack([record["features"] for record in records]), epoch=0)
    trainer = ControllerTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=args.device,
        logger=logger,
        scheduler=scheduler,
    )
    history = trainer.train(
        train_dataset,
        val_dataset,
        epochs=train_cfg.get("epochs", config["training"]["epochs"]),
        early_stop_patience=train_cfg.get("early_stop_patience", 10),
        batch_size=train_cfg.get("batch_size", config["training"]["batch_size"]),
        checkpoint_dir=args.checkpoint_dir,
    )

    data_out = Path(args.data_out)
    data_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        data_out,
        features=np.stack([record["features"] for record in records]),
        ds_expert=np.array([record["ds_expert"] for record in records], dtype=np.float32),
        y_restart=np.array([record["y_restart"] for record in records], dtype=np.int64),
    )

    history_path = Path(args.checkpoint_dir) / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    logger.close()
    print(f"epochs_run={len(history)} final_val={history[-1]['val']['loss']:.6f} records={len(records)}")


if __name__ == "__main__":
    main()
