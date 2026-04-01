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
from src.utils.contour_init import project_to_contour, sigma_min_at
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
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-attempts", type=int, default=32, help="Retry with new random matrix/start until a closed contour is found.")
    parser.add_argument(
        "--sample-mode",
        choices=["trained_epsilon", "point_sigma"],
        default="point_sigma",
        help="trained_epsilon: sample a random direction then project to the config epsilon contour; point_sigma: use epsilon=sigma_min(z0I-A) from a raw random point.",
    )
    parser.add_argument(
        "--min-step-size",
        type=float,
        default=None,
        help="Optional lower bound applied to controller-predicted step size during demo inference.",
    )
    parser.add_argument(
        "--radius-range",
        type=float,
        nargs=2,
        default=(0.15, 0.35),
        metavar=("R_MIN", "R_MAX"),
        help="Random radius range as a fraction of spectral scale around a random eigenvalue.",
    )
    parser.add_argument("--print-every", type=int, default=1, help="Print NN decisions every k tracking steps.")
    parser.add_argument("--quiet", action="store_true", help="Disable step-by-step console diagnostics.")
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


class DemoController:
    def __init__(self, base_controller: NNController, min_step_size: float):
        self.base_controller = base_controller
        self.min_step_size = float(min_step_size)

    def predict(self, state_np: np.ndarray) -> tuple[float, bool]:
        ds, need_restart = self.base_controller.predict(state_np)
        return max(float(ds), self.min_step_size), bool(need_restart)

    def predict_with_info(self, state_np: np.ndarray) -> tuple[float, bool, dict]:
        if hasattr(self.base_controller, "predict_with_info"):
            ds, need_restart, info = self.base_controller.predict_with_info(state_np)
        else:
            ds, need_restart = self.base_controller.predict(state_np)
            info = {}
        ds = max(float(ds), self.min_step_size)
        info = dict(info)
        info["min_step_size_applied"] = self.min_step_size
        return ds, bool(need_restart), info


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


def _format_complex(z: complex) -> str:
    return f"{float(np.real(z)):+.5f}{float(np.imag(z)):+.5f}j"


def make_step_printer(print_every: int):
    def _printer(info: dict) -> None:
        step = int(info["step"])
        if print_every > 1 and (step % print_every != 0):
            return
        restart_prob = info.get("controller_info", {}).get("restart_prob")
        restart_prob_str = "None" if restart_prob is None else f"{restart_prob:.4f}"
        print(
            f"  step={step:04d} "
            f"ds={info['ds']:.6f} "
            f"p_restart={restart_prob_str} "
            f"need_restart={int(info['need_restart'])} "
            f"applied_restart={int(info['applied_restart'])} "
            f"|dz|={info['step_distance']:.6f} "
            f"|z-z0|={info['distance_to_start']:.6f} "
            f"path={info['path_length']:.6f} "
            f"wind={info['winding_angle']:.4f} "
            f"z={_format_complex(info['z_next'])}"
        )

    return _printer


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
    base_controller = load_controller(args.checkpoint, config)
    min_step_size = float(args.min_step_size if args.min_step_size is not None else config["ode"]["initial_step_size"])
    controller = DemoController(base_controller, min_step_size=min_step_size)
    target_epsilon = float(config["ode"]["epsilon"])
    budget_multipliers = [1, 2, 4, 8]

    best_attempt = None
    for attempt_idx in range(args.max_attempts):
        A = generate_random_matrix(args.matrix_size, args.matrix_type, rng).astype(np.complex128)
        z_guess, anchor = choose_random_start(A, rng, tuple(args.radius_range))
        try:
            if args.sample_mode == "trained_epsilon":
                z0, epsilon = project_to_contour(A, target_epsilon, z_guess)
                epsilon = float(sigma_min_at(A, z0))
            else:
                z0 = z_guess
                epsilon = float(sigma_min_at(A, z0))
        except ValueError:
            continue
        ode = ManifoldODE(A, epsilon=epsilon, solver=solver)
        tracker = ContourTracker(
            A=A,
            epsilon=epsilon,
            ode_system=ode,
            controller=controller,
            fixed_step_size=config["ode"]["initial_step_size"],
            closure_tol=config["tracker"]["closure_tol"],
            min_steps_between_restarts=5,
        )
        if not args.quiet:
            print(
                f"attempt={attempt_idx + 1}/{args.max_attempts} "
                f"epsilon={epsilon:.6f} "
                f"z0={_format_complex(z0)} "
                f"anchor={_format_complex(anchor)}"
            )
        best_local = None
        for multiplier in budget_multipliers:
            step_budget = int(max(args.max_steps * multiplier, 64))
            if not args.quiet:
                print(f"  step_budget={step_budget}")
            result = tracker.track(
                z0=z0,
                max_steps=step_budget,
                step_callback=None if args.quiet else make_step_printer(max(args.print_every, 1)),
            )
            if not args.quiet:
                print(
                    f"  summary closed={int(result['closed'])} "
                    f"points={len(result['trajectory'])} "
                    f"restarts={len(result['restart_indices'])} "
                    f"closure_error={float(np.abs(result['trajectory'][-1] - result['trajectory'][0])):.6f} "
                    f"path={result['path_length']:.6f} "
                    f"wind={result['winding_angle']:.4f}"
                )
            local_attempt = {
                "attempt_index": attempt_idx,
                "step_budget": step_budget,
                "A": A,
                "z0": z0,
                "z_guess": z_guess,
                "anchor": anchor,
                "epsilon": epsilon,
                "result": result,
            }
            if best_local is None:
                best_local = local_attempt
            else:
                local_best_score = (
                    int(bool(best_local["result"]["closed"])),
                    abs(float(best_local["result"]["winding_angle"])),
                    float(best_local["result"]["path_length"]),
                    -float(np.abs(best_local["result"]["trajectory"][-1] - best_local["result"]["trajectory"][0])),
                )
                current_local_score = (
                    int(bool(result["closed"])),
                    abs(float(result["winding_angle"])),
                    float(result["path_length"]),
                    -float(np.abs(result["trajectory"][-1] - result["trajectory"][0])),
                )
                if current_local_score > local_best_score:
                    best_local = local_attempt
            if result["closed"]:
                break
        attempt = best_local
        if best_attempt is None:
            best_attempt = attempt
        else:
            best_score = (
                int(bool(best_attempt["result"]["closed"])),
                abs(float(best_attempt["result"]["winding_angle"])),
                float(best_attempt["result"]["path_length"]),
                -float(np.abs(best_attempt["result"]["trajectory"][-1] - best_attempt["result"]["trajectory"][0])),
            )
            current_score = (
                int(bool(attempt["result"]["closed"])),
                abs(float(attempt["result"]["winding_angle"])),
                float(attempt["result"]["path_length"]),
                -float(np.abs(attempt["result"]["trajectory"][-1] - attempt["result"]["trajectory"][0])),
            )
            if current_score > best_score:
                best_attempt = attempt
        if attempt["result"]["closed"]:
            best_attempt = attempt
            break

    if best_attempt is None or not best_attempt["result"]["closed"]:
        if best_attempt is not None and not args.quiet:
            best_result = best_attempt["result"]
            print(
                f"best_attempt_so_far attempt={best_attempt['attempt_index'] + 1} "
                f"step_budget={best_attempt['step_budget']} "
                f"closed={int(best_result['closed'])} "
                f"points={len(best_result['trajectory'])} "
                f"restarts={len(best_result['restart_indices'])} "
                f"closure_error={float(np.abs(best_result['trajectory'][-1] - best_result['trajectory'][0])):.6f} "
                f"path={best_result['path_length']:.6f} "
                f"wind={best_result['winding_angle']:.4f}"
            )
        raise RuntimeError(
            f"Failed to obtain a closed contour after {args.max_attempts} attempts. "
            "Increase --max-attempts or inspect the controller predictions."
        )

    A = best_attempt["A"]
    z0 = best_attempt["z0"]
    z_guess = best_attempt["z_guess"]
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
        "sample_mode": args.sample_mode,
        "attempt_index": int(best_attempt["attempt_index"]),
        "max_attempts": int(args.max_attempts),
        "step_budget_used": int(best_attempt["step_budget"]),
        "min_step_size": float(min_step_size),
        "epsilon": float(epsilon),
        "z_guess": [float(np.real(z_guess)), float(np.imag(z_guess))],
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
