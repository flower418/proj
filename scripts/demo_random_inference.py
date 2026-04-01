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
from src.utils.contour_init import sigma_min_at
from src.utils.visualization import plot_trajectory


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a random matrix, choose a random contour start z0, define epsilon from z0, and run full tracking."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained controller checkpoint.")
    parser.add_argument("--output-dir", default="results/random_inference", help="Directory for matrix, plot, and summary outputs.")
    parser.add_argument("--matrix-size", type=int, default=20)
    parser.add_argument("--matrix-type", choices=["complex", "real", "hermitian"], default="complex")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-attempts", type=int, default=8, help="Retry with new random matrix/start until a closed contour is found.")
    parser.add_argument(
        "--radius-range",
        type=float,
        nargs=2,
        default=(0.15, 0.35),
        metavar=("R_MIN", "R_MAX"),
        help="Random radius range as a fraction of spectral scale around a random eigenvalue.",
    )
    return parser.parse_args()


def generate_random_matrix(n: int, matrix_type: str, rng: np.random.Generator) -> np.ndarray:
    if matrix_type == "complex":
        return rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    if matrix_type == "real":
        return rng.standard_normal((n, n))
    if matrix_type == "hermitian":
        base = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        return (base + base.conj().T) / 2.0
    raise ValueError(f"Unsupported matrix type: {matrix_type}")


def choose_random_start(A: np.ndarray, rng: np.random.Generator, radius_range: tuple[float, float]) -> tuple[complex, complex]:
    eigvals = np.linalg.eigvals(np.asarray(A, dtype=np.complex128))
    anchor = complex(eigvals[int(rng.integers(len(eigvals)))])
    spectral_scale = max(np.ptp(np.real(eigvals)), np.ptp(np.imag(eigvals)), 1.0)
    r_min, r_max = radius_range
    if not (0.0 < r_min < r_max):
        raise ValueError("radius_range must satisfy 0 < R_MIN < R_MAX.")
    radius = rng.uniform(r_min, r_max) * spectral_scale
    angle = rng.uniform(0.0, 2.0 * np.pi)
    z0 = anchor + radius * np.exp(1j * angle)
    return z0, anchor


def load_controller(checkpoint_path: str, config: dict) -> NNController:
    controller = NNController(
        hidden_dims=config["controller"]["hidden_dims"],
        dropout=config["controller"]["dropout"],
        norm_type=config["controller"]["norm_type"],
        step_size_min=config["controller"]["step_size_min"],
        step_size_max=config["controller"]["step_size_max"],
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    controller.load_state_dict(checkpoint["model_state_dict"])
    controller.eval()
    return controller


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    config = load_yaml_config(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_out = output_dir / "random_matrix.npy"
    plot_out = output_dir / "tracked_contour.png"
    result_out = output_dir / "tracking_summary.json"

    solver = PseudoinverseSolver(
        method=config["solver"]["method"],
        tol=config["solver"]["tol"],
        max_iter=config["solver"]["max_iter"],
    )
    controller = load_controller(args.checkpoint, config)

    best_attempt = None
    for attempt_idx in range(args.max_attempts):
        A = generate_random_matrix(args.matrix_size, args.matrix_type, rng).astype(np.complex128)
        z0, anchor = choose_random_start(A, rng, tuple(args.radius_range))
        epsilon = sigma_min_at(A, z0)
        ode = ManifoldODE(A, epsilon=epsilon, solver=solver)
        tracker = ContourTracker(
            A=A,
            epsilon=epsilon,
            ode_system=ode,
            controller=controller,
            fixed_step_size=config["ode"]["initial_step_size"],
            closure_tol=config["tracker"]["closure_tol"],
        )
        result = tracker.track(z0=z0, max_steps=args.max_steps)
        attempt = {
            "attempt_index": attempt_idx,
            "A": A,
            "z0": z0,
            "anchor": anchor,
            "epsilon": epsilon,
            "result": result,
        }
        if best_attempt is None:
            best_attempt = attempt
        else:
            best_score = (
                int(bool(best_attempt["result"]["closed"])),
                abs(float(best_attempt["result"]["winding_angle"])),
                float(best_attempt["result"]["path_length"]),
                -float(best_attempt["result"]["closure_error"]) if "closure_error" in best_attempt["result"] else 0.0,
            )
            current_score = (
                int(bool(result["closed"])),
                abs(float(result["winding_angle"])),
                float(result["path_length"]),
                -float(np.abs(result["trajectory"][-1] - result["trajectory"][0])),
            )
            if current_score > best_score:
                best_attempt = attempt
        if result["closed"]:
            best_attempt = attempt
            break

    if best_attempt is None or not best_attempt["result"]["closed"]:
        raise RuntimeError(
            f"Failed to obtain a closed contour after {args.max_attempts} attempts. "
            "Increase --max-attempts or inspect the controller predictions."
        )

    A = best_attempt["A"]
    z0 = best_attempt["z0"]
    anchor = best_attempt["anchor"]
    epsilon = float(best_attempt["epsilon"])
    result = best_attempt["result"]
    np.save(matrix_out, A)

    plot_trajectory(
        trajectory=result["trajectory"],
        restart_indices=result["restart_indices"],
        step_sizes=result["step_sizes"],
        A=A,
        epsilon=epsilon,
        title=f"Random Demo Contour (n={args.matrix_size}, epsilon={epsilon:.4g}, z0={z0:.4f})",
        save_path=str(plot_out),
    )

    summary = {
        "checkpoint": args.checkpoint,
        "matrix_path": str(matrix_out),
        "plot_path": str(plot_out),
        "matrix_size": args.matrix_size,
        "matrix_type": args.matrix_type,
        "seed": args.seed,
        "attempt_index": int(best_attempt["attempt_index"]),
        "max_attempts": int(args.max_attempts),
        "epsilon": float(epsilon),
        "z0": [float(np.real(z0)), float(np.imag(z0))],
        "anchor_eigenvalue": [float(np.real(anchor)), float(np.imag(anchor))],
        "tracked_points": int(len(result["trajectory"])),
        "num_restarts": int(len(result["restart_indices"])),
        "closure_error": float(np.abs(result["trajectory"][-1] - result["trajectory"][0])),
        "closed": bool(result["closed"]),
        "path_length": float(result["path_length"]),
        "max_distance_from_start": float(result["max_distance_from_start"]),
        "winding_angle": float(result["winding_angle"]),
    }

    with result_out.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
