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
from src.nn.controller import NNController, build_controller_from_checkpoint
from src.nn.inference_controller import AdaptiveInferenceController
from src.utils.config import load_yaml_config
from src.utils.contour_init import project_to_contour, sigma_min_at
from src.utils.run_logging import StepDiagnosticsCollector, RunLogger, format_nn_step, make_step_callback
from src.utils.visualization import plot_trajectory


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a random matrix, sample a random complex-plane point, define epsilon from that point, and track the full contour with NN + ODE."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained controller checkpoint.")
    parser.add_argument("--output-dir", default="results/random_inference", help="Directory for matrix, plot, and summary outputs.")
    parser.add_argument("--matrix-size", type=int, default=20)
    parser.add_argument("--matrix-type", choices=["complex", "real", "hermitian"], default="complex")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=32, help="Retry with new random matrix/start until a closed contour is found.")
    parser.add_argument(
        "--point-sampler",
        choices=["spectral_box", "around_eigenvalue"],
        default="spectral_box",
        help="How to sample the random complex-plane point used to define the contour.",
    )
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
        help="Only used with --point-sampler around_eigenvalue. Random radius range as a fraction of spectral scale around a random eigenvalue.",
    )
    parser.add_argument(
        "--box-padding",
        type=float,
        default=0.25,
        help="Only used with --point-sampler spectral_box. Padding ratio applied to the eigenvalue bounding box before sampling a random point.",
    )
    parser.add_argument("--print-every", type=int, default=1, help="Print NN decisions every k tracking steps.")
    parser.add_argument("--quiet", action="store_true", help="Disable step-by-step console diagnostics.")
    parser.add_argument("--require-closed", action="store_true", help="Raise an error if no closed contour is found.")
    parser.add_argument("--log-dir", default=None, help="Directory for runtime logs. Defaults to <output-dir>/logs.")
    parser.add_argument("--restart-threshold", type=float, default=0.9, help="Controller restart probability threshold during inference.")
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


def choose_point_around_random_eigenvalue(
    A: np.ndarray,
    rng: np.random.Generator,
    radius_range: tuple[float, float],
) -> tuple[complex, complex]:
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


def choose_point_in_spectral_box(
    A: np.ndarray,
    rng: np.random.Generator,
    padding: float,
) -> tuple[complex, complex]:
    eigvals = np.linalg.eigvals(np.asarray(A, dtype=np.complex128))
    x_min = float(np.min(np.real(eigvals)))
    x_max = float(np.max(np.real(eigvals)))
    y_min = float(np.min(np.imag(eigvals)))
    y_max = float(np.max(np.imag(eigvals)))
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    x_pad = max(float(padding), 0.0) * x_span
    y_pad = max(float(padding), 0.0) * y_span
    z_random = complex(
        rng.uniform(x_min - x_pad, x_max + x_pad),
        rng.uniform(y_min - y_pad, y_max + y_pad),
    )
    nearest = complex(eigvals[int(np.argmin(np.abs(eigvals - z_random)))])
    return z_random, nearest


def load_controller(checkpoint_path: str, config: dict) -> NNController:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    controller = build_controller_from_checkpoint(
        checkpoint,
        config["controller"],
        input_dim=int(config["controller"].get("input_dim", 7)),
    )
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
            f"raw_ds={info.get('raw_ds', info['ds']):.6f} "
            f"ds={info['ds']:.6f} "
            f"p_restart={restart_prob_str} "
            f"need_restart={int(info['need_restart'])} "
            f"applied_restart={int(info['applied_restart'])} "
            f"applied_projection={int(info.get('applied_projection', False))} "
            f"backtracks={int(info.get('backtracks', 0))} "
            f"|dz|={info['step_distance']:.6f} "
            f"|z-z0|={info['distance_to_start']:.6f} "
            f"path={info['path_length']:.6f} "
            f"wind={info['winding_angle']:.4f} "
            f"sigma_err={info.get('sigma_error', float('nan')):.6e} "
            f"z={_format_complex(info['z_next'])}"
        )

    return _printer


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_root = Path(args.log_dir) if args.log_dir is not None else output_dir / "logs"
    matrix_out = output_dir / "random_matrix.npy"
    plot_out = output_dir / "tracked_contour.png"
    result_out = output_dir / "tracking_summary.json"

    with RunLogger(log_root, run_name="demo_random_inference") as run_logger:
        rng = np.random.default_rng(args.seed)
        config = load_yaml_config(args.config)
        run_logger.write_json("run_config.json", {"args": vars(args), "config": config})

        solver = PseudoinverseSolver(
            method=config["solver"]["method"],
            tol=config["solver"]["tol"],
            max_iter=config["solver"]["max_iter"],
        )
        base_controller = load_controller(args.checkpoint, config)
        min_step_size = float(
            args.min_step_size
            if args.min_step_size is not None
            else config["controller"]["step_size_min"]
        )
        controller = AdaptiveInferenceController(
            base_controller,
            min_step_size=min_step_size,
            max_step_size=config["controller"].get("step_size_max"),
            restart_threshold=float(args.restart_threshold),
        )
        target_epsilon = float(config["ode"]["epsilon"])
        base_step_budget = int(args.max_steps if args.max_steps is not None else config["tracker"]["max_steps"])
        budget_multipliers = [1, 2, 4, 8]

        best_attempt = None
        for attempt_idx in range(args.max_attempts):
            A = generate_random_matrix(args.matrix_size, args.matrix_type, rng).astype(np.complex128)
            if args.point_sampler == "spectral_box":
                z_random, anchor = choose_point_in_spectral_box(A, rng, padding=args.box_padding)
            else:
                z_random, anchor = choose_point_around_random_eigenvalue(A, rng, tuple(args.radius_range))
            try:
                if args.sample_mode == "trained_epsilon":
                    z0, epsilon = project_to_contour(A, target_epsilon, z_random)
                    epsilon = float(sigma_min_at(A, z0))
                else:
                    z0 = z_random
                    epsilon = float(sigma_min_at(A, z0))
            except ValueError:
                run_logger.log(f"attempt={attempt_idx + 1}/{args.max_attempts} projection failed, resampling")
                continue

            run_logger.log(
                f"attempt={attempt_idx + 1}/{args.max_attempts} "
                f"epsilon={epsilon:.6f} random_point={_format_complex(z_random)} "
                f"start_point={_format_complex(z0)} nearest_eig={_format_complex(anchor)}"
            )
            ode = ManifoldODE(A, epsilon=epsilon, solver=solver)
            tracker = ContourTracker(
                A=A,
                epsilon=epsilon,
                ode_system=ode,
                controller=controller,
                fixed_step_size=config["ode"]["initial_step_size"],
                closure_tol=config["tracker"]["closure_tol"],
                min_steps_between_restarts=5,
                integration_method="tangent",
                projection_defer_factor=4.0,
                projection_defer_distance_ratio=0.08,
                max_deferred_projection_steps=6,
                exact_triplet_refresh_interval=2,
                approx_triplet_sigma_tol=5.0e-4,
                approx_triplet_residual_tol=1.0e-3,
            )

            best_local = None
            for multiplier in budget_multipliers:
                step_budget = int(max(base_step_budget * multiplier, 64))
                run_logger.log(f"attempt={attempt_idx + 1} step_budget={step_budget}")
                collector = StepDiagnosticsCollector(label=f"attempt_{attempt_idx + 1}_budget_{step_budget}")
                step_callback = make_step_callback(
                    run_logger=run_logger,
                    collector=collector,
                    jsonl_filename="tracking_steps.jsonl",
                    formatter=lambda info, label=f"a{attempt_idx + 1}": format_nn_step(info, label=label),
                    print_every=0 if args.quiet else max(args.print_every, 1),
                    info_transform=lambda info, attempt=attempt_idx, budget=step_budget: {
                        **info,
                        "attempt_index": int(attempt),
                        "step_budget": int(budget),
                    },
                )
                result = tracker.track(z0=z0, max_steps=step_budget, step_callback=step_callback)
                diagnostics = collector.summary()
                closure_error = float(np.abs(result["trajectory"][-1] - result["trajectory"][0]))
                attempt_summary = {
                    "attempt_index": int(attempt_idx),
                    "step_budget": int(step_budget),
                    "closed": bool(result["closed"]),
                    "tracked_points": int(len(result["trajectory"])),
                    "num_restarts": int(len(result["restart_indices"])),
                    "num_projections": int(len(result.get("projection_indices", []))),
                    "closure_error": closure_error,
                    "path_length": float(result["path_length"]),
                    "winding_angle": float(result["winding_angle"]),
                    "diagnostics": diagnostics,
                }
                run_logger.append_jsonl("attempts.jsonl", attempt_summary)
                run_logger.log(
                    f"attempt={attempt_idx + 1} step_budget={step_budget} "
                    f"closed={int(result['closed'])} points={len(result['trajectory'])} "
                    f"restarts={len(result['restart_indices'])} projections={len(result.get('projection_indices', []))} "
                    f"closure_error={closure_error:.6f} path={result['path_length']:.6f} wind={result['winding_angle']:.4f}"
                )

                local_attempt = {
                    "attempt_index": attempt_idx,
                    "step_budget": step_budget,
                    "A": A,
                    "z0": z0,
                    "z_random": z_random,
                    "anchor": anchor,
                    "epsilon": epsilon,
                    "result": result,
                    "diagnostics": diagnostics,
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
                        -closure_error,
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
            if best_attempt is not None:
                best_result = best_attempt["result"]
                run_logger.log(
                    f"best_attempt_so_far attempt={best_attempt['attempt_index'] + 1} "
                    f"step_budget={best_attempt['step_budget']} closed={int(best_result['closed'])} "
                    f"points={len(best_result['trajectory'])} restarts={len(best_result['restart_indices'])} "
                    f"closure_error={float(np.abs(best_result['trajectory'][-1] - best_result['trajectory'][0])):.6f} "
                    f"path={best_result['path_length']:.6f} wind={best_result['winding_angle']:.4f}"
                )
            if best_attempt is None:
                raise RuntimeError(
                    f"Failed to obtain any valid tracking attempt after {args.max_attempts} attempts."
                )
            if args.require_closed:
                raise RuntimeError(
                    f"Failed to obtain a closed contour after {args.max_attempts} attempts. "
                    "Increase --max-attempts or inspect the controller predictions."
                )

        A = best_attempt["A"]
        z0 = best_attempt["z0"]
        z_random = best_attempt["z_random"]
        anchor = best_attempt["anchor"]
        epsilon = float(best_attempt["epsilon"])
        result = best_attempt["result"]
        diagnostics = best_attempt["diagnostics"]
        diagnostics_path = run_logger.write_json("best_attempt_diagnostics_summary.json", diagnostics)
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
            "algorithm": "nn_plus_ode",
            "checkpoint": args.checkpoint,
            "matrix_path": str(matrix_out),
            "plot_path": str(plot_out),
            "log_dir": str(run_logger.log_dir),
            "run_log_path": str(run_logger.run_log_path),
            "step_log_path": str(run_logger.log_dir / "tracking_steps.jsonl"),
            "attempt_log_path": str(run_logger.log_dir / "attempts.jsonl"),
            "diagnostics_path": str(diagnostics_path),
            "matrix_size": args.matrix_size,
            "matrix_type": args.matrix_type,
            "seed": args.seed,
            "point_sampler": args.point_sampler,
            "sample_mode": args.sample_mode,
            "attempt_index": int(best_attempt["attempt_index"]),
            "max_attempts": int(args.max_attempts),
            "step_budget_used": int(best_attempt["step_budget"]),
            "min_step_size": float(min_step_size),
            "restart_threshold": float(args.restart_threshold),
            "epsilon": float(epsilon),
            "random_point": [float(np.real(z_random)), float(np.imag(z_random))],
            "sigma_at_random_point": float(sigma_min_at(A, z_random)),
            "start_point": [float(np.real(z0)), float(np.imag(z0))],
            "nearest_eigenvalue": [float(np.real(anchor)), float(np.imag(anchor))],
            "tracked_points": int(len(result["trajectory"])),
            "num_restarts": int(len(result["restart_indices"])),
            "closure_error": float(np.abs(result["trajectory"][-1] - result["trajectory"][0])),
            "closed": bool(result["closed"]),
            "path_length": float(result["path_length"]),
            "max_distance_from_start": float(result["max_distance_from_start"]),
            "winding_angle": float(result["winding_angle"]),
            "num_projections": int(len(result.get("projection_indices", []))),
            "diagnostics": diagnostics,
        }

        with result_out.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        run_logger.log(f"demo finished summary={result_out}")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
