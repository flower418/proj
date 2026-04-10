from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import torch

import _bootstrap  # noqa: F401

from src.baselines import NewtonPredictorCorrectorTracker
from src.core.contour_tracker import FAST_TANGENT_TRACKER_KWARGS, ContourTracker
from src.core.manifold_ode import ManifoldODE
from src.core.pseudoinverse import PseudoinverseSolver
from src.nn.controller import NNController, build_controller_from_checkpoint
from src.nn.inference_controller import AdaptiveInferenceController
from src.utils.config import load_yaml_config
from src.utils.contour_compare import contour_distance_metrics
from src.utils.demo_sampling import build_random_matrix, sample_random_contour_start
from src.utils.run_logging import RunLogger, format_nn_step


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark NN+ODE contour tracking against a Newton predictor-corrector baseline on the same random matrix and point."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="results/nn_vs_newton")
    parser.add_argument("--matrix-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=16000)
    parser.add_argument("--nn-min-step-size", type=float, default=None)
    parser.add_argument("--restart-threshold", type=float, default=0.9)
    parser.add_argument("--baseline-initial-step-size", type=float, default=1e-2)
    parser.add_argument("--baseline-min-step-size", type=float, default=1e-6)
    parser.add_argument("--baseline-max-step-size", type=float, default=1e-1)
    parser.add_argument("--baseline-corrector-tol", type=float, default=1e-10)
    parser.add_argument("--baseline-max-corrector-iters", type=int, default=8)
    parser.add_argument("--baseline-max-step-halvings", type=int, default=8)
    parser.add_argument("--log-dir", default=None, help="Directory for runtime logs. Defaults to <output-dir>/logs.")
    return parser.parse_args()


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


def make_nn_console_callback(print_every: int = 20):
    print_every = max(int(print_every), 1)

    def _callback(info: dict) -> None:
        step = int(info.get("step", 0)) + 1
        if step % print_every != 0:
            return
        payload = dict(info)
        payload["step"] = step
        print(format_nn_step(payload, label="nn"), flush=True)

    return _callback


def run_nn_benchmark(
    *,
    A: np.ndarray,
    epsilon: float,
    z0: complex,
    checkpoint: str,
    config: dict,
    max_steps: int,
    nn_min_step_size: float | None,
    restart_threshold: float,
    summary_path: Path,
) -> dict:
    solver = PseudoinverseSolver(
        method=config["solver"]["method"],
        tol=config["solver"]["tol"],
        max_iter=config["solver"]["max_iter"],
    )
    controller = load_controller(checkpoint, config)
    nn_controller = AdaptiveInferenceController(
        controller,
        min_step_size=float(nn_min_step_size if nn_min_step_size is not None else config["controller"]["step_size_min"]),
        max_step_size=config["controller"].get("step_size_max"),
        restart_threshold=float(restart_threshold),
    )

    tracker = ContourTracker(
        A=A,
        epsilon=epsilon,
        ode_system=ManifoldODE(A, epsilon=epsilon, solver=solver),
        controller=nn_controller,
        fixed_step_size=config["ode"]["initial_step_size"],
        closure_tol=config["tracker"]["closure_tol"],
        min_step_size=config["ode"]["min_step_size"],
        **FAST_TANGENT_TRACKER_KWARGS,
    )
    step_callback = make_nn_console_callback(print_every=20)
    t0 = time.perf_counter()
    result = tracker.track(z0=z0, max_steps=max_steps, step_callback=step_callback)
    elapsed = float(time.perf_counter() - t0)
    summary = summarize_tracker_result(result)
    summary.update(
        {
            "elapsed_seconds": elapsed,
            "total_time_seconds": elapsed,
            "num_restarts": int(len(result.get("restart_indices", []))),
            "num_projections": int(len(result.get("projection_indices", []))),
        }
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return {
        "result": result,
        "elapsed_seconds": elapsed,
        "summary": summary,
        "summary_path": str(summary_path),
    }


def run_newton_benchmark(
    *,
    A: np.ndarray,
    epsilon: float,
    z0: complex,
    closure_tol: float,
    max_steps: int,
    initial_step_size: float,
    min_step_size: float,
    max_step_size: float,
    corrector_tol: float,
    max_corrector_iters: int,
    max_step_halvings: int,
) -> dict:
    tracker = NewtonPredictorCorrectorTracker(
        A=A,
        epsilon=epsilon,
        initial_step_size=initial_step_size,
        min_step_size=min_step_size,
        max_step_size=max_step_size,
        corrector_tol=corrector_tol,
        max_corrector_iters=max_corrector_iters,
        max_step_halvings=max_step_halvings,
        closure_tol=closure_tol,
    )
    t0 = time.perf_counter()
    result = tracker.track(z0=z0, max_steps=max_steps)
    elapsed = float(time.perf_counter() - t0)
    summary = summarize_tracker_result(result)
    summary.update(
        {
            "elapsed_seconds": elapsed,
            "total_time_seconds": elapsed,
            "mean_corrector_iterations": float(result.get("mean_corrector_iterations", 0.0)),
            "mean_predictor_halvings": float(result.get("mean_predictor_halvings", 0.0)),
            "mean_line_search_backtracks": float(result.get("mean_line_search_backtracks", 0.0)),
            "failure_reason": result.get("failure_reason"),
        }
    )
    return {
        "result": result,
        "elapsed_seconds": elapsed,
        "summary": summary,
    }


def sample_marker_points(trajectory: np.ndarray, count: int, phase: float = 0.0) -> np.ndarray:
    traj = np.asarray(trajectory, dtype=np.complex128)
    if len(traj) == 0:
        return traj
    if len(traj) <= count:
        return traj
    phase = float(np.clip(phase, 0.0, 0.95))
    positions = np.linspace(phase, len(traj) - 1 - phase, count)
    indices = np.unique(np.clip(np.round(positions).astype(int), 0, len(traj) - 1))
    return traj[indices]


def save_comparison_plot(
    A: np.ndarray,
    epsilon: float,
    z_random: complex,
    z0: complex,
    matrix_type: str,
    nn_result: dict,
    baseline_result: dict,
    nn_elapsed: float,
    baseline_elapsed: float,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 8.6))

    combined = [np.asarray(nn_result["trajectory"], dtype=np.complex128), np.asarray(baseline_result["trajectory"], dtype=np.complex128)]
    merged = np.concatenate(combined)
    real_margin = max(np.ptp(np.real(merged)) * 0.12, 1.0)
    imag_margin = max(np.ptp(np.imag(merged)) * 0.12, 1.0)
    ax.set_xlim(np.real(merged).min() - real_margin, np.real(merged).max() + real_margin)
    ax.set_ylim(np.imag(merged).min() - imag_margin, np.imag(merged).max() + imag_margin)

    nn_traj = np.asarray(nn_result["trajectory"], dtype=np.complex128)
    base_traj = np.asarray(baseline_result["trajectory"], dtype=np.complex128)
    nn_line, = ax.plot(
        np.real(nn_traj),
        np.imag(nn_traj),
        color="#0f4c81",
        linewidth=2.8,
        linestyle="-",
        alpha=0.90,
        label="NN + ODE",
        zorder=6,
    )
    nn_line.set_path_effects(
        [
            pe.Stroke(linewidth=nn_line.get_linewidth() + 1.4, foreground="white", alpha=0.88),
            pe.Normal(),
        ]
    )

    nn_markers = sample_marker_points(nn_traj, count=12, phase=0.0)
    base_markers = sample_marker_points(base_traj, count=12, phase=0.35)
    ax.scatter(
        np.real(base_markers),
        np.imag(base_markers),
        s=22,
        marker="^",
        facecolors=(1.0, 0.92, 0.84, 0.18),
        edgecolors=(0.85, 0.37, 0.01, 0.55),
        linewidths=0.8,
        zorder=6,
    )
    ax.scatter(
        np.real(nn_markers),
        np.imag(nn_markers),
        s=24,
        marker="o",
        facecolors=(0.06, 0.30, 0.51, 0.95),
        edgecolors="white",
        linewidths=0.8,
        zorder=7,
    )
    base_line, = ax.plot(
        np.real(base_traj),
        np.imag(base_traj),
        color="#d95f02",
        linewidth=3.6,
        alpha=0.58,
        linestyle=(0, (7.5, 3.5)),
        solid_capstyle="round",
        dash_capstyle="round",
        label="Newton PC",
        zorder=8,
    )
    base_line.set_path_effects(
        [
            pe.Stroke(linewidth=base_line.get_linewidth() + 1.0, foreground="white", alpha=0.75),
            pe.Normal(),
        ]
    )
    ax.scatter(
        np.real(z_random), np.imag(z_random), c="#2a9d8f", s=115, marker="X", edgecolors="black", linewidths=0.7, label="Random Point", zorder=10
    )
    ax.scatter(np.real(z0), np.imag(z0), c="white", s=92, marker="o", edgecolors="black", linewidths=1.1, label="Start", zorder=10)
    ax.scatter(np.real(nn_traj[-1]), np.imag(nn_traj[-1]), c="#0f4c81", s=82, marker="s", edgecolors="white", linewidths=0.9, label="NN End", zorder=10)
    ax.scatter(np.real(base_traj[-1]), np.imag(base_traj[-1]), c="#d95f02", s=84, marker="D", edgecolors="white", linewidths=0.9, label="Newton End", zorder=10)

    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.set_title(
        f"NN + ODE vs Newton Predictor-Corrector\nmatrix={matrix_type}, epsilon={epsilon:.4g}",
        loc="left",
        fontsize=11,
        pad=18,
    )
    ax.grid(True, alpha=0.3)
    time_lines = [
        f"NN + ODE: {nn_elapsed:.3f}s",
        f"Newton PC: {baseline_elapsed:.3f}s",
        f"NN closed: {int(bool(nn_result.get('closed', False)))}",
        f"Newton closed: {int(bool(baseline_result.get('closed', False)))}",
    ]
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
    handles, labels = ax.get_legend_handles_labels()
    fig.subplots_adjust(top=0.9, right=0.8)
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=9,
            frameon=True,
            borderaxespad=0.0,
        )
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 0.96))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_root = Path(args.log_dir) if args.log_dir is not None else output_dir / "logs"

    with RunLogger(log_root, run_name="benchmark_nn_vs_newton") as run_logger:
        config = load_yaml_config(args.config)

        run_logger.write_json("run_config.json", {"args": vars(args), "config": config})
        run_logger.log(f"benchmark start log_dir={run_logger.log_dir}")

        rng = np.random.default_rng(args.seed)
        prep_t0 = time.perf_counter()
        matrix_type, A = build_random_matrix(args.matrix_size, rng)
        z0, epsilon, z_random, anchor = sample_random_contour_start(A=A, rng=rng)
        epsilon_compute_seconds = float(time.perf_counter() - prep_t0)

        run_logger.log(
            f"prepared instance n={args.matrix_size} type={matrix_type} "
            f"epsilon={epsilon:.6f} z_random={z_random} z0={z0}"
        )
        run_logger.log("running NN + ODE first, then Newton baseline")
        benchmark_t0 = time.perf_counter()
        nn_run = run_nn_benchmark(
            A=A,
            epsilon=epsilon,
            z0=z0,
            checkpoint=args.checkpoint,
            config=config,
            max_steps=args.max_steps,
            nn_min_step_size=args.nn_min_step_size,
            restart_threshold=args.restart_threshold,
            summary_path=run_logger.log_dir / "nn" / "summary.json",
        )
        baseline_run = run_newton_benchmark(
            A=A,
            epsilon=epsilon,
            z0=z0,
            closure_tol=config["tracker"]["closure_tol"],
            max_steps=args.max_steps,
            initial_step_size=args.baseline_initial_step_size,
            min_step_size=args.baseline_min_step_size,
            max_step_size=args.baseline_max_step_size,
            corrector_tol=args.baseline_corrector_tol,
            max_corrector_iters=args.baseline_max_corrector_iters,
            max_step_halvings=args.baseline_max_step_halvings,
        )
        benchmark_elapsed = float(time.perf_counter() - benchmark_t0)
        run_logger.log(
            f"sequential benchmark done wall_clock={benchmark_elapsed:.3f}s "
            f"nn={nn_run['elapsed_seconds']:.3f}s newton={baseline_run['elapsed_seconds']:.3f}s"
        )

        nn_result = nn_run["result"]
        baseline_result = baseline_run["result"]
        nn_elapsed = float(nn_run["elapsed_seconds"])
        baseline_elapsed = float(baseline_run["elapsed_seconds"])
        nn_summary = dict(nn_run["summary"])
        baseline_summary = dict(baseline_run["summary"])

        comparison = {
            "nn_vs_newton": contour_distance_metrics(nn_result["trajectory"], baseline_result["trajectory"]),
        }

        matrix_out = output_dir / "random_matrix.npy"
        plot_out = output_dir / "comparison_plot.png"
        summary_out = output_dir / "comparison_summary.json"
        traj_out = output_dir / "trajectories.npz"

        np.save(matrix_out, A)
        np.savez(
            traj_out,
            nn_trajectory=np.asarray(nn_result["trajectory"], dtype=np.complex128),
            baseline_trajectory=np.asarray(baseline_result["trajectory"], dtype=np.complex128),
        )
        save_comparison_plot(
            A=A,
            epsilon=epsilon,
            z_random=z_random,
            z0=z0,
            matrix_type=matrix_type,
            nn_result=nn_result,
            baseline_result=baseline_result,
            nn_elapsed=nn_elapsed,
            baseline_elapsed=baseline_elapsed,
            save_path=plot_out,
        )

        summary = {
            "algorithm": "nn_plus_ode_vs_newton_predictor_corrector",
            "checkpoint": args.checkpoint,
            "matrix_path": str(matrix_out),
            "plot_path": str(plot_out),
            "trajectories_path": str(traj_out),
            "log_dir": str(run_logger.log_dir),
            "run_log_path": str(run_logger.run_log_path),
            "execution_mode": "sequential",
            "wall_clock_seconds": benchmark_elapsed,
            "total_time_seconds": benchmark_elapsed,
            "nn_summary_path": nn_run["summary_path"],
            "matrix_size": args.matrix_size,
            "matrix_type": matrix_type,
            "seed": args.seed,
            "sampling_strategy": "random_point_sigma",
            "restart_threshold": float(args.restart_threshold),
            "epsilon": float(epsilon),
            "epsilon_compute_seconds": epsilon_compute_seconds,
            "random_point": [float(np.real(z_random)), float(np.imag(z_random))],
            "start_point": [float(np.real(z0)), float(np.imag(z0))],
            "nearest_eigenvalue": [float(np.real(anchor)), float(np.imag(anchor))],
            "nn_plus_ode": nn_summary,
            "newton_predictor_corrector": baseline_summary,
            "comparison": comparison,
        }
        with summary_out.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        run_logger.log(f"benchmark finished summary={summary_out}")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
