from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # noqa: F401

from src.baselines import NewtonPredictorCorrectorTracker
from src.utils.contour_init import project_to_contour, sigma_min_at
from src.utils.demo_sampling import generate_random_matrix, sample_random_point
from src.utils.visualization import plot_pseudospectrum_background


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Newton predictor-corrector contour baseline on a random matrix and random point.")
    parser.add_argument("--output-dir", default="results/newton_baseline")
    parser.add_argument("--matrix-size", type=int, default=20)
    parser.add_argument("--matrix-type", choices=["complex", "real", "hermitian"], default="complex")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=16000)
    parser.add_argument("--point-sampler", choices=["spectral_box", "around_eigenvalue"], default="spectral_box")
    parser.add_argument("--sample-mode", choices=["trained_epsilon", "point_sigma"], default="point_sigma")
    parser.add_argument("--epsilon-value", type=float, default=0.1)
    parser.add_argument("--radius-range", type=float, nargs=2, default=(0.15, 0.35), metavar=("R_MIN", "R_MAX"))
    parser.add_argument("--box-padding", type=float, default=0.25)
    parser.add_argument("--initial-step-size", type=float, default=1e-2)
    parser.add_argument("--min-step-size", type=float, default=1e-6)
    parser.add_argument("--max-step-size", type=float, default=1e-1)
    parser.add_argument("--corrector-tol", type=float, default=1e-10)
    parser.add_argument("--max-corrector-iters", type=int, default=8)
    parser.add_argument("--max-step-halvings", type=int, default=8)
    parser.add_argument("--closure-tol", type=float, default=1e-3)
    return parser.parse_args()


def save_plot(
    A: np.ndarray,
    epsilon: float,
    z_random: complex,
    trajectory: np.ndarray,
    elapsed_seconds: float,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    traj = np.asarray(trajectory, dtype=np.complex128)
    real_margin = max(np.ptp(np.real(traj)) * 0.12, 1.0)
    imag_margin = max(np.ptp(np.imag(traj)) * 0.12, 1.0)
    ax.set_xlim(np.real(traj).min() - real_margin, np.real(traj).max() + real_margin)
    ax.set_ylim(np.imag(traj).min() - imag_margin, np.imag(traj).max() + imag_margin)
    plot_pseudospectrum_background(A, epsilon, ax, resolution=120, alpha=0.16)
    ax.plot(np.real(traj), np.imag(traj), color="tab:orange", linewidth=2.0, label="Newton PC", zorder=4)
    ax.scatter(np.real(z_random), np.imag(z_random), c="green", s=110, marker="o", edgecolors="black", linewidths=0.6, label="Random Point", zorder=6)
    ax.scatter(np.real(traj[0]), np.imag(traj[0]), c="blue", s=70, marker="o", label="Start", zorder=6)
    ax.scatter(np.real(traj[-1]), np.imag(traj[-1]), c="red", s=70, marker="s", label="End", zorder=6)
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.set_title(f"Newton Predictor-Corrector Baseline (epsilon={epsilon:.4g})", loc="left", fontsize=11, pad=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=9, frameon=True)
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
    ax.set_aspect("equal")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
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
    if args.sample_mode == "trained_epsilon":
        z0, epsilon = project_to_contour(A, args.epsilon_value, z_random)
        epsilon = float(sigma_min_at(A, z0))
    else:
        z0 = z_random
        epsilon = float(sigma_min_at(A, z0))
    epsilon_compute_seconds = float(time.perf_counter() - prep_t0)

    tracker = NewtonPredictorCorrectorTracker(
        A=A,
        epsilon=epsilon,
        initial_step_size=args.initial_step_size,
        min_step_size=args.min_step_size,
        max_step_size=args.max_step_size,
        corrector_tol=args.corrector_tol,
        max_corrector_iters=args.max_corrector_iters,
        max_step_halvings=args.max_step_halvings,
        closure_tol=args.closure_tol,
    )
    t0 = time.perf_counter()
    result = tracker.track(z0=z0, max_steps=args.max_steps)
    elapsed_seconds = float(time.perf_counter() - t0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "random_matrix.npy", A)
    np.savez(output_dir / "trajectory.npz", trajectory=np.asarray(result["trajectory"], dtype=np.complex128))
    save_plot(
        A=A,
        epsilon=epsilon,
        z_random=z_random,
        trajectory=np.asarray(result["trajectory"], dtype=np.complex128),
        elapsed_seconds=elapsed_seconds,
        save_path=output_dir / "newton_baseline.png",
    )

    summary = {
        "algorithm": "newton_predictor_corrector",
        "matrix_path": str(output_dir / "random_matrix.npy"),
        "plot_path": str(output_dir / "newton_baseline.png"),
        "trajectory_path": str(output_dir / "trajectory.npz"),
        "matrix_size": args.matrix_size,
        "matrix_type": args.matrix_type,
        "seed": args.seed,
        "point_sampler": args.point_sampler,
        "sample_mode": args.sample_mode,
        "epsilon_compute_seconds": epsilon_compute_seconds,
        "elapsed_seconds": elapsed_seconds,
        "epsilon": float(epsilon),
        "random_point": [float(np.real(z_random)), float(np.imag(z_random))],
        "sigma_at_random_point": float(sigma_min_at(A, z_random)),
        "start_point": [float(np.real(z0)), float(np.imag(z0))],
        "nearest_eigenvalue": [float(np.real(anchor)), float(np.imag(anchor))],
        "tracked_points": int(len(result["trajectory"])),
        "closed": bool(result["closed"]),
        "closure_error": float(abs(result["trajectory"][-1] - result["trajectory"][0])),
        "path_length": float(result["path_length"]),
        "winding_angle": float(result["winding_angle"]),
        "mean_corrector_iterations": float(result.get("mean_corrector_iterations", 0.0)),
        "mean_predictor_halvings": float(result.get("mean_predictor_halvings", 0.0)),
        "mean_line_search_backtracks": float(result.get("mean_line_search_backtracks", 0.0)),
        "failure_reason": result.get("failure_reason"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
