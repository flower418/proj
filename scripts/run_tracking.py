from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401

from src.core.contour_tracker import ContourTracker
from src.core.manifold_ode import ManifoldODE
from src.core.pseudoinverse import PseudoinverseSolver
from src.nn.controller import NNController
from src.utils.config import load_yaml_config
from src.utils.visualization import plot_trajectory


def parse_args():
    parser = argparse.ArgumentParser(description="Run contour tracking inference.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--matrix-path", default=None, help="Path to .npy or .npz containing matrix A.")
    parser.add_argument("--matrix-size", type=int, default=12, help="Only used together with --demo-random.")
    parser.add_argument("--demo-random", action="store_true", help="Use a random demo matrix instead of a supplied matrix.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--plot-out", default=None)
    parser.add_argument("--result-out", default=None, help="Optional .json summary output.")
    parser.add_argument("--z0-real", type=float, default=0.5)
    parser.add_argument("--z0-imag", type=float, default=0.2)
    return parser.parse_args()


def load_matrix(matrix_path: str) -> np.ndarray:
    path = Path(matrix_path)
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data = np.load(path)
        if "A" in data:
            return data["A"]
        if len(data.files) == 1:
            return data[data.files[0]]
        raise ValueError("NPZ matrix file must contain key 'A' or a single array.")
    raise ValueError("Unsupported matrix file format. Use .npy or .npz.")


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    if args.matrix_path is None and not args.demo_random:
        raise ValueError("Provide --matrix-path for real inference, or use --demo-random for a toy random-matrix demo.")
    np.random.seed(args.seed)
    if args.matrix_path is not None:
        A = np.asarray(load_matrix(args.matrix_path), dtype=np.complex128)
    else:
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
    closure_error = float(np.abs(result["trajectory"][-1] - result["trajectory"][0]))
    if args.plot_out is not None:
        plot_path = Path(args.plot_out)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_trajectory(
            trajectory=result["trajectory"],
            restart_indices=result["restart_indices"],
            step_sizes=result["step_sizes"],
            A=A,
            epsilon=config["ode"]["epsilon"],
            title=f"Contour Tracking from z0={complex(args.z0_real, args.z0_imag)}",
            save_path=str(plot_path),
        )
    summary = {
        "tracked_points": int(len(result["trajectory"])),
        "num_restarts": int(len(result["restart_indices"])),
        "closure_error": closure_error,
        "start_point": [float(np.real(result["trajectory"][0])), float(np.imag(result["trajectory"][0]))],
        "end_point": [float(np.real(result["trajectory"][-1])), float(np.imag(result["trajectory"][-1]))],
        "mode": "user_matrix" if args.matrix_path is not None else "demo_random",
    }
    if args.result_out is not None:
        out_path = Path(args.result_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
