from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import _bootstrap  # noqa: F401

from src.baselines import NewtonPredictorCorrectorTracker
from src.core.contour_tracker import ContourTracker
from src.core.manifold_ode import ManifoldODE
from src.core.pseudoinverse import PseudoinverseSolver
from src.nn.controller import NNController, build_controller_from_checkpoint
from src.utils.config import load_yaml_config
from src.utils.contour_compare import contour_distance_metrics, resample_curve_by_arclength
from src.utils.contour_init import project_to_contour, sigma_min_at
from src.utils.demo_sampling import generate_random_matrix, sample_random_point
from src.utils.visualization import plot_trajectory
from src.utils.visualization import plot_pseudospectrum_background


class TimedDemoController:
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark NN+ODE contour tracking against a Newton predictor-corrector baseline on the same random matrix and point."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="results/nn_vs_newton")
    parser.add_argument("--matrix-size", type=int, default=20)
    parser.add_argument("--matrix-type", choices=["complex", "real", "hermitian"], default="complex")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=16000)
    parser.add_argument("--point-sampler", choices=["spectral_box", "around_eigenvalue"], default="spectral_box")
    parser.add_argument("--sample-mode", choices=["trained_epsilon", "point_sigma"], default="point_sigma")
    parser.add_argument("--radius-range", type=float, nargs=2, default=(0.15, 0.35), metavar=("R_MIN", "R_MAX"))
    parser.add_argument("--box-padding", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--nn-min-step-size", type=float, default=None)
    parser.add_argument("--baseline-initial-step-size", type=float, default=1e-2)
    parser.add_argument("--baseline-min-step-size", type=float, default=1e-6)
    parser.add_argument("--baseline-max-step-size", type=float, default=1e-1)
    parser.add_argument("--baseline-corrector-tol", type=float, default=1e-10)
    parser.add_argument("--baseline-max-corrector-iters", type=int, default=8)
    parser.add_argument("--baseline-max-step-halvings", type=int, default=8)
    parser.add_argument("--reference-step-size", type=float, default=2e-3)
    parser.add_argument("--reference-corrector-tol", type=float, default=1e-12)
    parser.add_argument("--reference-max-steps", type=int, default=40000)
    return parser.parse_args()


def load_controller(checkpoint_path: str, config: dict, device: torch.device) -> NNController:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    controller = build_controller_from_checkpoint(
        checkpoint,
        config["controller"],
        input_dim=int(config["controller"].get("input_dim", 7)),
    )
    controller.load_state_dict(checkpoint["model_state_dict"])
    controller = controller.to(device)
    controller.eval()
    return controller


def summarize_tracker_result(result: dict) -> dict:
    trajectory = np.asarray(result["trajectory"], dtype=np.complex128)
    step_sizes = np.asarray(result.get("step_sizes", []), dtype=np.float64)
    closure_error = float(abs(trajectory[-1] - trajectory[0])) if len(trajectory) >= 2 else 0.0
    return {
        "closed": bool(result.get("closed", False)),
        "tracked_points": int(len(trajectory)),
        "closure_error": closure_error,
        "path_length": float(result.get("path_length", 0.0)),
        "max_distance_from_start": float(result.get("max_distance_from_start", 0.0)),
        "winding_angle": float(result.get("winding_angle", 0.0)),
        "mean_step_size": float(np.mean(step_sizes)) if step_sizes.size > 0 else 0.0,
        "min_step_size": float(np.min(step_sizes)) if step_sizes.size > 0 else 0.0,
        "max_step_size": float(np.max(step_sizes)) if step_sizes.size > 0 else 0.0,
    }


def compute_sigma_error_stats(
    A: np.ndarray,
    epsilon: float,
    trajectory: np.ndarray,
    num_samples: int = 256,
) -> dict[str, float]:
    sampled = resample_curve_by_arclength(trajectory, num_points=min(num_samples, max(len(trajectory), 1)))
    sigma_errors = [abs(sigma_min_at(A, complex(z)) - epsilon) for z in sampled]
    sigma_errors = np.asarray(sigma_errors, dtype=np.float64)
    return {
        "mean_sigma_error": float(np.mean(sigma_errors)) if sigma_errors.size > 0 else 0.0,
        "max_sigma_error": float(np.max(sigma_errors)) if sigma_errors.size > 0 else 0.0,
    }


def save_comparison_plot(
    A: np.ndarray,
    epsilon: float,
    z_random: complex,
    nn_result: dict,
    baseline_result: dict,
    reference_result: dict | None,
    nn_elapsed: float,
    baseline_elapsed: float,
    reference_elapsed: float | None,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    combined = [np.asarray(nn_result["trajectory"], dtype=np.complex128), np.asarray(baseline_result["trajectory"], dtype=np.complex128)]
    if reference_result is not None:
        combined.append(np.asarray(reference_result["trajectory"], dtype=np.complex128))
    merged = np.concatenate(combined)
    real_margin = max(np.ptp(np.real(merged)) * 0.12, 1.0)
    imag_margin = max(np.ptp(np.imag(merged)) * 0.12, 1.0)
    ax.set_xlim(np.real(merged).min() - real_margin, np.real(merged).max() + real_margin)
    ax.set_ylim(np.imag(merged).min() - imag_margin, np.imag(merged).max() + imag_margin)
    plot_pseudospectrum_background(A, epsilon, ax, resolution=120, alpha=0.16)

    nn_traj = np.asarray(nn_result["trajectory"], dtype=np.complex128)
    base_traj = np.asarray(baseline_result["trajectory"], dtype=np.complex128)
    ax.plot(np.real(nn_traj), np.imag(nn_traj), color="tab:blue", linewidth=2.0, label="NN + ODE", zorder=4)
    ax.plot(np.real(base_traj), np.imag(base_traj), color="tab:orange", linewidth=2.0, label="Newton PC", zorder=4)
    if reference_result is not None:
        ref_traj = np.asarray(reference_result["trajectory"], dtype=np.complex128)
        ax.plot(np.real(ref_traj), np.imag(ref_traj), color="black", linewidth=1.4, linestyle="--", alpha=0.8, label="Reference", zorder=3)

    ax.scatter(np.real(z_random), np.imag(z_random), c="green", s=110, marker="o", edgecolors="black", linewidths=0.6, label="Random Point", zorder=6)
    ax.scatter(np.real(nn_traj[-1]), np.imag(nn_traj[-1]), c="tab:blue", s=70, marker="s", label="NN End", zorder=6)
    ax.scatter(np.real(base_traj[-1]), np.imag(base_traj[-1]), c="tab:orange", s=70, marker="D", label="Newton End", zorder=6)

    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.set_title(f"NN + ODE vs Newton Predictor-Corrector (epsilon={epsilon:.4g})", loc="left", fontsize=11, pad=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=9, frameon=True)
    time_lines = [
        f"NN + ODE: {nn_elapsed:.3f}s",
        f"Newton PC: {baseline_elapsed:.3f}s",
    ]
    if reference_elapsed is not None:
        time_lines.append(f"Reference: {reference_elapsed:.3f}s")
    ax.text(
        0.02,
        0.98,
        "\n".join(time_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        zorder=10,
    )
    ax.set_aspect("equal")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_single_method_plot(
    A: np.ndarray,
    epsilon: float,
    trajectory: np.ndarray,
    restart_indices: list[int],
    step_sizes: np.ndarray | list[float],
    title: str,
    elapsed_seconds: float,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_trajectory(
        trajectory=np.asarray(trajectory, dtype=np.complex128),
        restart_indices=restart_indices,
        step_sizes=np.asarray(step_sizes, dtype=np.float64) if len(step_sizes) > 0 else None,
        A=A,
        epsilon=epsilon,
        ax=ax,
        title=title,
    )
    ax.text(
        0.02,
        0.98,
        f"Time: {elapsed_seconds:.3f}s",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        zorder=10,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前 PyTorch 未检测到可用 GPU。")

    rng = np.random.default_rng(args.seed)
    A = generate_random_matrix(args.matrix_size, args.matrix_type, rng).astype(np.complex128)
    z_random, anchor = sample_random_point(
        A=A,
        rng=rng,
        point_sampler=args.point_sampler,
        radius_range=tuple(args.radius_range),
        box_padding=args.box_padding,
    )

    prep_t0 = time.perf_counter()
    target_epsilon = float(config["ode"]["epsilon"])
    if args.sample_mode == "trained_epsilon":
        z0, epsilon = project_to_contour(A, target_epsilon, z_random)
        epsilon = float(sigma_min_at(A, z0))
    else:
        z0 = z_random
        epsilon = float(sigma_min_at(A, z0))
    epsilon_compute_seconds = float(time.perf_counter() - prep_t0)

    solver = PseudoinverseSolver(
        method=config["solver"]["method"],
        tol=config["solver"]["tol"],
        max_iter=config["solver"]["max_iter"],
    )
    controller = load_controller(args.checkpoint, config, device=device)
    nn_controller = TimedDemoController(
        controller,
        min_step_size=float(args.nn_min_step_size if args.nn_min_step_size is not None else config["controller"]["step_size_min"]),
    )

    nn_tracker = ContourTracker(
        A=A,
        epsilon=epsilon,
        ode_system=ManifoldODE(A, epsilon=epsilon, solver=solver),
        controller=nn_controller,
        fixed_step_size=config["ode"]["initial_step_size"],
        closure_tol=config["tracker"]["closure_tol"],
        min_step_size=config["ode"]["min_step_size"],
    )
    nn_t0 = time.perf_counter()
    nn_result = nn_tracker.track(z0=z0, max_steps=args.max_steps)
    nn_elapsed = float(time.perf_counter() - nn_t0)

    baseline_tracker = NewtonPredictorCorrectorTracker(
        A=A,
        epsilon=epsilon,
        initial_step_size=args.baseline_initial_step_size,
        min_step_size=args.baseline_min_step_size,
        max_step_size=args.baseline_max_step_size,
        corrector_tol=args.baseline_corrector_tol,
        max_corrector_iters=args.baseline_max_corrector_iters,
        max_step_halvings=args.baseline_max_step_halvings,
        closure_tol=config["tracker"]["closure_tol"],
    )
    baseline_t0 = time.perf_counter()
    baseline_result = baseline_tracker.track(z0=z0, max_steps=args.max_steps)
    baseline_elapsed = float(time.perf_counter() - baseline_t0)

    reference_tracker = NewtonPredictorCorrectorTracker(
        A=A,
        epsilon=epsilon,
        initial_step_size=args.reference_step_size,
        min_step_size=args.reference_step_size,
        max_step_size=args.reference_step_size,
        corrector_tol=args.reference_corrector_tol,
        max_corrector_iters=max(args.baseline_max_corrector_iters + 4, 12),
        max_step_halvings=0,
        closure_tol=config["tracker"]["closure_tol"],
    )
    reference_t0 = time.perf_counter()
    reference_result = reference_tracker.track(z0=z0, max_steps=args.reference_max_steps)
    reference_elapsed = float(time.perf_counter() - reference_t0)
    if not reference_result["closed"]:
        reference_result = None

    nn_summary = summarize_tracker_result(nn_result)
    nn_summary.update(compute_sigma_error_stats(A, epsilon, nn_result["trajectory"]))
    nn_summary.update(
        {
            "elapsed_seconds": nn_elapsed,
            "num_restarts": int(len(nn_result.get("restart_indices", []))),
            "num_projections": int(len(nn_result.get("projection_indices", []))),
        }
    )

    baseline_summary = summarize_tracker_result(baseline_result)
    baseline_summary.update(compute_sigma_error_stats(A, epsilon, baseline_result["trajectory"]))
    baseline_summary.update(
        {
            "elapsed_seconds": baseline_elapsed,
            "mean_corrector_iterations": float(baseline_result.get("mean_corrector_iterations", 0.0)),
            "mean_predictor_halvings": float(baseline_result.get("mean_predictor_halvings", 0.0)),
            "mean_line_search_backtracks": float(baseline_result.get("mean_line_search_backtracks", 0.0)),
            "failure_reason": baseline_result.get("failure_reason"),
        }
    )

    comparison = {
        "nn_vs_newton": contour_distance_metrics(nn_result["trajectory"], baseline_result["trajectory"]),
    }
    if reference_result is not None:
        reference_summary = summarize_tracker_result(reference_result)
        reference_summary.update(compute_sigma_error_stats(A, epsilon, reference_result["trajectory"]))
        reference_summary.update(
            {
                "elapsed_seconds": reference_elapsed,
                "mode": "high_precision_newton_reference",
            }
        )
        comparison["nn_vs_reference"] = contour_distance_metrics(nn_result["trajectory"], reference_result["trajectory"])
        comparison["newton_vs_reference"] = contour_distance_metrics(baseline_result["trajectory"], reference_result["trajectory"])
    else:
        reference_summary = {
            "closed": False,
            "elapsed_seconds": reference_elapsed,
            "mode": "high_precision_newton_reference",
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_out = output_dir / "random_matrix.npy"
    plot_out = output_dir / "comparison_plot.png"
    nn_plot_out = output_dir / "nn_only_plot.png"
    baseline_plot_out = output_dir / "newton_only_plot.png"
    summary_out = output_dir / "comparison_summary.json"
    traj_out = output_dir / "trajectories.npz"

    np.save(matrix_out, A)
    np.savez(
        traj_out,
        nn_trajectory=np.asarray(nn_result["trajectory"], dtype=np.complex128),
        baseline_trajectory=np.asarray(baseline_result["trajectory"], dtype=np.complex128),
        reference_trajectory=None if reference_result is None else np.asarray(reference_result["trajectory"], dtype=np.complex128),
    )
    save_comparison_plot(
        A=A,
        epsilon=epsilon,
        z_random=z_random,
        nn_result=nn_result,
        baseline_result=baseline_result,
        reference_result=reference_result,
        nn_elapsed=nn_elapsed,
        baseline_elapsed=baseline_elapsed,
        reference_elapsed=reference_elapsed if reference_result is not None else None,
        save_path=plot_out,
    )
    save_single_method_plot(
        A=A,
        epsilon=epsilon,
        trajectory=np.asarray(nn_result["trajectory"], dtype=np.complex128),
        restart_indices=list(nn_result.get("restart_indices", [])),
        step_sizes=np.asarray(nn_result.get("step_sizes", []), dtype=np.float64),
        title=f"NN + ODE Contour (epsilon={epsilon:.4g})",
        elapsed_seconds=nn_elapsed,
        save_path=nn_plot_out,
    )
    save_single_method_plot(
        A=A,
        epsilon=epsilon,
        trajectory=np.asarray(baseline_result["trajectory"], dtype=np.complex128),
        restart_indices=[],
        step_sizes=np.asarray(baseline_result.get("step_sizes", []), dtype=np.float64),
        title=f"Newton Predictor-Corrector Contour (epsilon={epsilon:.4g})",
        elapsed_seconds=baseline_elapsed,
        save_path=baseline_plot_out,
    )

    summary = {
        "algorithm": "nn_plus_ode_vs_newton_predictor_corrector",
        "checkpoint": args.checkpoint,
        "matrix_path": str(matrix_out),
        "plot_path": str(plot_out),
        "nn_plot_path": str(nn_plot_out),
        "baseline_plot_path": str(baseline_plot_out),
        "trajectories_path": str(traj_out),
        "matrix_size": args.matrix_size,
        "matrix_type": args.matrix_type,
        "seed": args.seed,
        "point_sampler": args.point_sampler,
        "sample_mode": args.sample_mode,
        "training_epsilon": float(target_epsilon),
        "epsilon": float(epsilon),
        "epsilon_compute_seconds": epsilon_compute_seconds,
        "random_point": [float(np.real(z_random)), float(np.imag(z_random))],
        "sigma_at_random_point": float(sigma_min_at(A, z_random)),
        "start_point": [float(np.real(z0)), float(np.imag(z0))],
        "nearest_eigenvalue": [float(np.real(anchor)), float(np.imag(anchor))],
        "nn_plus_ode": nn_summary,
        "newton_predictor_corrector": baseline_summary,
        "reference": reference_summary,
        "comparison": comparison,
    }
    with summary_out.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
