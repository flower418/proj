from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.core.contour_tracker import ContourTracker
from src.core.manifold_ode import ManifoldODE
from src.core.pseudoinverse import PseudoinverseSolver
from src.nn.controller import NNController
from src.utils.config import load_yaml_config
from src.utils.visualization import plot_trajectory


def parse_args():
    parser = argparse.ArgumentParser(description="Run contour tracking inference.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--matrix-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--plot-out", default=None)
    parser.add_argument("--z0-real", type=float, default=0.5)
    parser.add_argument("--z0-imag", type=float, default=0.2)
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
    controller = None
    if args.checkpoint is not None:
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
    ode = ManifoldODE(A, epsilon=config["ode"]["epsilon"], solver=solver)
    tracker = ContourTracker(
        A=A,
        epsilon=config["ode"]["epsilon"],
        ode_system=ode,
        controller=controller,
        fixed_step_size=config["ode"]["initial_step_size"],
        closure_tol=config["tracker"]["closure_tol"],
    )
    result = tracker.track(z0=complex(args.z0_real, args.z0_imag), max_steps=args.max_steps)
    if args.plot_out is not None:
        plot_path = Path(args.plot_out)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_trajectory(
            trajectory=result["trajectory"],
            restart_indices=result["restart_indices"],
            step_sizes=result["step_sizes"],
            A=A,
            epsilon=config["ode"]["epsilon"],
            title=f"Neural Controller Tracking (ε={config['ode']['epsilon']})",
            save_path=str(plot_path),
        )
    print(f"tracked_points={len(result['trajectory'])} restarts={len(result['restart_indices'])}")


if __name__ == "__main__":
    main()
