from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401

from src.core.contour_tracker import ContourTracker
from src.nn.controller import build_controller_from_checkpoint
from src.nn.inference_controller import AdaptiveInferenceController
from src.utils.demo_sampling import build_visual_demo_matrix, sample_near_eigen_contour_start
from src.utils.contour_init import auto_select_contour_start, project_to_contour, sigma_min_at
from src.utils.run_logging import StepDiagnosticsCollector, RunLogger, format_nn_step, make_step_callback
from src.utils.visualization import plot_trajectory


DEFAULT_CONTROLLER_CONFIG = {
    "input_dim": 6,
    "hidden_dims": [128, 128, 64],
    "dropout": 0.05,
    "norm_type": "layernorm",
    "activation": "silu",
    "head_hidden_dim": 64,
    "step_size_min": 1.0e-4,
    "step_size_max": 0.1,
}

DEFAULT_ODE = {
    "epsilon": 0.1,
    "initial_step_size": 1.0e-2,
    "min_step_size": 1.0e-6,
}

DEFAULT_TRACKER = {
    "max_steps": 4000,
    "closure_tol": 1.0e-3,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run contour tracking inference.")
    parser.add_argument("--matrix-path", default=None, help="Path to .npy or .npz containing matrix A.")
    parser.add_argument("--matrix-size", type=int, default=50, help="Only used together with --demo-random.")
    parser.add_argument("--demo-random", action="store_true", help="Use a random demo matrix instead of a supplied matrix.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--plot-out", default=None)
    parser.add_argument("--result-out", default=None, help="Optional .json summary output.")
    parser.add_argument("--epsilon", type=float, default=None, help="Override the epsilon level from config.")
    parser.add_argument("--z0-real", type=float, default=None, help="Real part of a user-supplied initial guess in the complex plane.")
    parser.add_argument("--z0-imag", type=float, default=None, help="Imag part of a user-supplied initial guess in the complex plane.")
    parser.add_argument(
        "--auto-start",
        choices=["rightmost", "leftmost", "topmost", "bottommost"],
        default="rightmost",
        help="Automatically choose a start point from the contour component around an extreme eigenvalue.",
    )
    parser.add_argument(
        "--auto-angle-offset",
        type=float,
        default=0.0,
        help="Rotate the automatic start ray by this many radians.",
    )
    parser.add_argument(
        "--near-eigen-contour",
        action="store_true",
        help="Choose a contour defined by a random point near a selected anchor eigenvalue.",
    )
    parser.add_argument(
        "--near-eigen-gap-ratio",
        type=float,
        default=0.04,
        help="Radius as a fraction of the nearest eigenvalue gap when --near-eigen-contour is enabled.",
    )
    parser.add_argument("--log-dir", default=None, help="Directory for runtime logs. Defaults near the output files.")
    parser.add_argument("--print-every", type=int, default=1, help="Print one step log every k steps. All steps are still saved to JSONL.")
    parser.add_argument("--quiet", action="store_true", help="Disable step-by-step console diagnostics while still saving logs to files.")
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
    if args.log_dir is not None:
        log_root = Path(args.log_dir)
    elif args.result_out is not None:
        log_root = Path(args.result_out).resolve().parent / "logs"
    elif args.plot_out is not None:
        log_root = Path(args.plot_out).resolve().parent / "logs"
    else:
        log_root = Path("results/run_tracking/logs")

    with RunLogger(log_root, run_name="run_tracking") as run_logger:
        run_logger.write_json(
            "run_config.json",
            {
                "args": vars(args),
                "defaults": {
                    "controller": DEFAULT_CONTROLLER_CONFIG,
                    "ode": DEFAULT_ODE,
                    "tracker": DEFAULT_TRACKER,
                },
            },
        )

        epsilon = float(args.epsilon if args.epsilon is not None else DEFAULT_ODE["epsilon"])
        max_steps = int(args.max_steps if args.max_steps is not None else DEFAULT_TRACKER["max_steps"])
        if args.matrix_path is None and not args.demo_random:
            raise ValueError("Provide --matrix-path for real inference, or use --demo-random for a toy random-matrix demo.")
        rng = np.random.default_rng(args.seed)
        if args.matrix_path is not None:
            A = np.asarray(load_matrix(args.matrix_path), dtype=np.complex128)
            matrix_type = "user_matrix"
        else:
            matrix_type, A = build_visual_demo_matrix(args.matrix_size, rng)

        start_mode = "auto"
        anchor_eigenvalue = None
        auto_start_radius = None
        near_eigen_angle = None
        z_input = None
        sigma_at_input = None
        if args.z0_real is not None or args.z0_imag is not None:
            if args.z0_real is None or args.z0_imag is None:
                raise ValueError("Provide both --z0-real and --z0-imag, or provide neither and use automatic start selection.")
            start_mode = "user_guess"
            z_input = complex(args.z0_real, args.z0_imag)
            sigma_at_input = sigma_min_at(A, z_input)
            z0, sigma_at_start = project_to_contour(A, epsilon, z_input)
        else:
            if args.near_eigen_contour:
                z0, epsilon, _, anchor_eigenvalue, sample_meta = sample_near_eigen_contour_start(
                    A,
                    rng=rng,
                    which=args.auto_start,
                    gap_ratio_range=(args.near_eigen_gap_ratio, args.near_eigen_gap_ratio),
                    fallback_radius_ratio_range=(args.near_eigen_gap_ratio, args.near_eigen_gap_ratio),
                )
                auto_start_radius = float(sample_meta.get("sampling_radius", 0.0))
                near_eigen_angle = sample_meta.get("sampling_angle")
                sigma_at_start = float(epsilon)
                start_mode = "auto_near_eigen"
            else:
                z0, sigma_at_start, anchor_eigenvalue = auto_select_contour_start(
                    A,
                    epsilon,
                    which=args.auto_start,
                    angle_offset=args.auto_angle_offset,
                )

        run_logger.log(
            f"tracking start mode={start_mode} epsilon={epsilon:.6f} max_steps={max_steps} "
            f"matrix_shape={A.shape} matrix_type={matrix_type} log_dir={run_logger.log_dir}"
        )

        controller = None
        if args.checkpoint is not None:
            checkpoint = torch.load(args.checkpoint, map_location="cpu")
            base_controller = build_controller_from_checkpoint(
                checkpoint,
                DEFAULT_CONTROLLER_CONFIG,
                input_dim=int(DEFAULT_CONTROLLER_CONFIG["input_dim"]),
            )
            base_controller.load_state_dict(checkpoint["model_state_dict"])
            base_controller.eval()
            controller = AdaptiveInferenceController(
                base_controller,
                min_step_size=float(DEFAULT_CONTROLLER_CONFIG["step_size_min"]),
                max_step_size=DEFAULT_CONTROLLER_CONFIG.get("step_size_max"),
            )

        collector = StepDiagnosticsCollector(label="run_tracking")
        step_callback = make_step_callback(
            run_logger=run_logger,
            collector=collector,
            jsonl_filename="tracking_steps.jsonl",
            formatter=lambda info: format_nn_step(info, label="tracking"),
            print_every=0 if args.quiet else max(args.print_every, 1),
        )

        tracker = ContourTracker(
            A=A,
            epsilon=epsilon,
            controller=controller,
            fixed_step_size=DEFAULT_ODE["initial_step_size"],
            closure_tol=DEFAULT_TRACKER["closure_tol"],
        )
        result = tracker.track(z0=z0, max_steps=max_steps, step_callback=step_callback)
        diagnostics = collector.summary()
        diagnostics_path = run_logger.write_json("tracking_diagnostics_summary.json", diagnostics)
        closure_error = float(np.abs(result["trajectory"][-1] - result["trajectory"][0]))

        if args.plot_out is not None:
            plot_path = Path(args.plot_out)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plot_trajectory(
                trajectory=result["trajectory"],
                step_sizes=result["step_sizes"],
                A=A,
                epsilon=epsilon,
                title=f"Contour Tracking (epsilon={epsilon:.4g}, start={start_mode}, z0={z0:.4f})",
                save_path=str(plot_path),
            )
        summary = {
            "tracked_points": int(len(result["trajectory"])),
            "num_projections": int(len(result.get("projection_indices", []))),
            "closure_error": closure_error,
            "closed": bool(result.get("closed", False)),
            "path_length": float(result.get("path_length", 0.0)),
            "max_distance_from_start": float(result.get("max_distance_from_start", 0.0)),
            "winding_angle": float(result.get("winding_angle", 0.0)),
            "epsilon": epsilon,
            "max_steps": max_steps,
            "matrix_type": matrix_type,
            "start_mode": start_mode,
            "input_point": None if z_input is None else [float(np.real(z_input)), float(np.imag(z_input))],
            "projected_start_point": [float(np.real(result["trajectory"][0])), float(np.imag(result["trajectory"][0]))],
            "end_point": [float(np.real(result["trajectory"][-1])), float(np.imag(result["trajectory"][-1]))],
            "sigma_at_input": sigma_at_input,
            "sigma_at_projected_start": sigma_at_start,
            "anchor_eigenvalue": None if anchor_eigenvalue is None else [float(np.real(anchor_eigenvalue)), float(np.imag(anchor_eigenvalue))],
            "auto_start_mode": args.auto_start if start_mode.startswith("auto") else None,
            "auto_angle_offset": args.auto_angle_offset if start_mode.startswith("auto") else None,
            "near_eigen_contour": bool(args.near_eigen_contour),
            "near_eigen_gap_ratio": float(args.near_eigen_gap_ratio) if args.near_eigen_contour else None,
            "auto_start_radius": auto_start_radius,
            "near_eigen_angle": near_eigen_angle,
            "mode": "user_matrix" if args.matrix_path is not None else "demo_random",
            "log_dir": str(run_logger.log_dir),
            "run_log_path": str(run_logger.run_log_path),
            "step_log_path": str(run_logger.log_dir / "tracking_steps.jsonl"),
            "diagnostics_path": str(diagnostics_path),
            "diagnostics": diagnostics,
        }
        if args.result_out is not None:
            out_path = Path(args.result_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)
        run_logger.log(
            f"tracking finished closed={int(summary['closed'])} points={summary['tracked_points']} "
            f"projections={summary['num_projections']}"
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
