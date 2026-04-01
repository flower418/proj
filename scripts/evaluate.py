from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.core.contour_tracker import ContourTracker
from src.core.manifold_ode import ManifoldODE
from src.core.pseudoinverse import PseudoinverseSolver
from src.nn.controller import NNController
from src.train.data_generator import ExpertDataGenerator, ExpertDataset
from src.train.logger import TrainingLogger
from src.train.trainer import ControllerTrainer
from src.nn.loss import ControllerLoss
from src.utils.metrics import contour_closure_error
from src.utils.config import load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained controller.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--matrix-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--z0-real", type=float, default=0.2)
    parser.add_argument("--z0-imag", type=float, default=0.1)
    parser.add_argument("--metrics-out", default="models/eval_metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    np.random.seed(args.seed)
    A = np.random.randn(args.matrix_size, args.matrix_size) + 1j * np.random.randn(args.matrix_size, args.matrix_size)
    solver = PseudoinverseSolver(
        method=config["solver"]["method"],
        tol=config["solver"]["tol"],
        max_iter=config["solver"]["max_iter"],
    )

    controller = NNController(
        hidden_dims=config["controller"]["hidden_dims"],
        dropout=config["controller"]["dropout"],
        norm_type=config["controller"]["norm_type"],
        step_size_min=config["controller"]["step_size_min"],
        step_size_max=config["controller"]["step_size_max"],
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    controller.load_state_dict(checkpoint["model_state_dict"])
    controller.eval()

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
    records = generator.generate_trajectory(complex(args.z0_real, args.z0_imag), max_steps=args.max_steps)
    dataset = ExpertDataset(records)
    logger = TrainingLogger(log_dir="logs", experiment_name="eval_tmp")
    trainer = ControllerTrainer(
        model=controller,
        loss_fn=ControllerLoss(
            lambda_step=config["training"]["lambda_step"],
            lambda_restart=config["training"]["lambda_restart"],
            alpha_restart=config["training"]["alpha_restart"],
            focal_gamma=config["training"]["focal_gamma"],
        ),
        optimizer=torch.optim.Adam(controller.parameters(), lr=config["training"]["learning_rate"]),
        device="cpu",
        logger=logger,
        scheduler=None,
    )
    metrics = trainer.evaluate(torch.utils.data.DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False))
    raw = metrics.pop("_raw", None)
    tracker = ContourTracker(
        A=A,
        epsilon=config["ode"]["epsilon"],
        ode_system=ManifoldODE(A, config["ode"]["epsilon"], solver=solver),
        controller=controller,
        fixed_step_size=config["ode"]["initial_step_size"],
        closure_tol=config["tracker"]["closure_tol"],
    )
    tracking_result = tracker.track(z0=complex(args.z0_real, args.z0_imag), max_steps=args.max_steps)
    metrics["closure_error"] = contour_closure_error(tracking_result["trajectory"])
    metrics["num_restarts"] = len(tracking_result["restart_indices"])
    if raw is not None:
        logger.log_confusion_matrix(raw["y_true"], (raw["y_prob"] >= 0.5).astype(np.int64), epoch=0)
        logger.log_prediction_scatter(raw["ds_pred"], raw["ds_true"], epoch=0)
    logger.close()
    out_path = Path(args.metrics_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
