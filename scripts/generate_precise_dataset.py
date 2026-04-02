from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import _bootstrap  # noqa: F401

from src.core.pseudoinverse import PseudoinverseSolver
from src.train.data_generator import ExpertDataGenerator
from src.utils.contour_init import project_to_contour, sigma_min_at
from src.utils.demo_sampling import generate_random_matrix, sample_random_point


def build_matrix(matrix_type: str, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if matrix_type == "complex":
        return generate_random_matrix(n, "complex", rng)
    if matrix_type == "real":
        return generate_random_matrix(n, "real", rng)
    if matrix_type == "hermitian":
        return generate_random_matrix(n, "hermitian", rng)
    if matrix_type == "ill_conditioned":
        u, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
        v, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
        sigma = np.geomspace(1.0, 1.0 / 1e6, n)
        return u @ np.diag(sigma) @ v.conj().T
    raise ValueError(f"Unsupported matrix_type: {matrix_type}")


def trajectory_closed(records: list[dict[str, Any]], z0: complex, closure_tol: float) -> bool:
    if not records:
        return False
    z_last = complex(records[-1]["z_next"])
    if len(records) < 16:
        return False
    return abs(z_last - z0) < closure_tol


def serialize_complex(z: complex) -> list[float]:
    return [float(np.real(z)), float(np.imag(z))]


def save_dataset(
    output_dir: Path,
    name: str,
    records: list[dict[str, Any]],
    trajectory_meta: list[dict[str, Any]],
    stats: dict[str, Any],
    rng: np.random.Generator,
) -> None:
    features = np.stack([np.asarray(r["features"], dtype=np.float32) for r in records])
    ds_expert = np.asarray([float(r["ds_expert"]) for r in records], dtype=np.float32)
    y_restart = np.asarray([int(r["y_restart"]) for r in records], dtype=np.int64)
    epsilon = np.asarray([float(r["epsilon"]) for r in records], dtype=np.float32)
    trajectory_id = np.asarray([int(r["trajectory_id"]) for r in records], dtype=np.int64)
    matrix_size = np.asarray([int(r["matrix_size"]) for r in records], dtype=np.int32)
    matrix_type = np.asarray([str(r["matrix_type"]) for r in records], dtype="<U32")
    point_sampler = np.asarray([str(r["point_sampler"]) for r in records], dtype="<U32")
    source = np.asarray([str(r.get("source", "expert")) for r in records], dtype="<U32")

    np.savez(
        output_dir / f"{name}.npz",
        features=features,
        ds_expert=ds_expert,
        y_restart=y_restart,
        epsilon=epsilon,
        trajectory_id=trajectory_id,
        matrix_size=matrix_size,
        matrix_type=matrix_type,
        point_sampler=point_sampler,
        source=source,
    )

    n = len(records)
    indices = rng.permutation(n)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    np.savez(
        output_dir / f"{name}_splits.npz",
        train_indices=indices[:train_end],
        val_indices=indices[train_end:val_end],
        test_indices=indices[val_end:],
    )
    if name == "dataset_full":
        np.savez(
            output_dir / "dataset_splits.npz",
            train_indices=indices[:train_end],
            val_indices=indices[train_end:val_end],
            test_indices=indices[val_end:],
        )

    with (output_dir / "dataset_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    with (output_dir / "trajectory_metadata.jsonl").open("w", encoding="utf-8") as fh:
        for item in trajectory_meta:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a precise contour-tracking dataset aligned with the random-point point_sigma task.")
    parser.add_argument("--target-samples", type=int, default=50000)
    parser.add_argument("--matrix-sizes", type=int, nargs="+", default=[20, 30, 50])
    parser.add_argument("--matrix-types", type=str, nargs="+", default=["complex", "real", "hermitian", "ill_conditioned"])
    parser.add_argument("--trajectories-per-type", type=int, default=4, help="How many random contours to sample per matrix before moving to the next matrix.")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--dagger-factor", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="data/prod50k_v2")
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-mode", choices=["point_sigma", "trained_epsilon"], default="point_sigma")
    parser.add_argument("--epsilon-value", type=float, default=0.1, help="Only used when --sample-mode trained_epsilon.")
    parser.add_argument("--point-samplers", type=str, nargs="+", default=["around_eigenvalue", "spectral_box"])
    parser.add_argument("--radius-range", type=float, nargs=2, default=(0.10, 0.30), metavar=("R_MIN", "R_MAX"))
    parser.add_argument("--box-padding", type=float, default=0.10)
    parser.add_argument("--expert-rtol", type=float, default=1e-9)
    parser.add_argument("--expert-atol", type=float, default=1e-9)
    parser.add_argument("--solver-tol", type=float, default=1e-10)
    parser.add_argument("--drift-threshold", type=float, default=1e-5)
    parser.add_argument("--closure-tol", type=float, default=1e-3)
    parser.add_argument("--base-step-size", type=float, default=1e-2)
    parser.add_argument("--max-step-size", type=float, default=1e-1)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--max-attempts-per-trajectory", type=int, default=8)
    parser.add_argument("--allow-open-trajectories", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    records: list[dict[str, Any]] = []
    trajectory_meta: list[dict[str, Any]] = []
    trajectory_id = 0
    matrix_counter = 0
    pairings = [(matrix_type, int(n)) for matrix_type in args.matrix_types for n in args.matrix_sizes]

    stats: dict[str, Any] = {
        "generator": "generate_precise_dataset",
        "sample_mode": args.sample_mode,
        "point_samplers": list(args.point_samplers),
        "target_samples": int(args.target_samples),
        "max_steps": int(args.max_steps),
        "dagger_factor": int(args.dagger_factor),
        "feature_dim": None,
        "num_samples": 0,
        "num_trajectories": 0,
        "num_closed_trajectories": 0,
        "num_open_trajectories": 0,
        "restart_samples": 0,
        "epsilon_min": None,
        "epsilon_max": None,
        "epsilon_mean": 0.0,
        "matrix_type_counts": {matrix_type: 0 for matrix_type in args.matrix_types},
        "matrix_size_counts": {str(n): 0 for n in args.matrix_sizes},
    }
    epsilon_values: list[float] = []

    while len(records) < args.target_samples:
        matrix_type, n = pairings[matrix_counter % len(pairings)]
        matrix_seed = args.seed + matrix_counter
        matrix_counter += 1

        A = build_matrix(matrix_type=matrix_type, n=n, seed=matrix_seed).astype(np.complex128)
        solver = PseudoinverseSolver(method="minres", tol=args.solver_tol, max_iter=1000)

        for local_traj_idx in range(args.trajectories_per_type):
            if len(records) >= args.target_samples:
                break

            accepted_records = None
            accepted_meta = None
            for attempt_idx in range(args.max_attempts_per_trajectory):
                sampler = str(args.point_samplers[(trajectory_id + attempt_idx) % len(args.point_samplers)])
                z_random, anchor = sample_random_point(
                    A=A,
                    rng=rng,
                    point_sampler=sampler,
                    radius_range=tuple(args.radius_range),
                    box_padding=args.box_padding,
                )
                if args.sample_mode == "trained_epsilon":
                    try:
                        z0, epsilon = project_to_contour(A, args.epsilon_value, z_random)
                    except ValueError:
                        continue
                    epsilon = float(sigma_min_at(A, z0))
                else:
                    z0 = complex(z_random)
                    epsilon = float(sigma_min_at(A, z0))

                generator = ExpertDataGenerator(
                    A=A,
                    epsilon=epsilon,
                    expert_tol=args.expert_rtol,
                    atol=args.expert_atol,
                    drift_threshold=args.drift_threshold,
                    base_step_size=args.base_step_size,
                    max_step_size=args.max_step_size,
                    closure_tol=args.closure_tol,
                    solver=solver,
                )
                trajectory = generator.generate_trajectory(z0=z0, max_steps=args.max_steps)
                if not trajectory:
                    continue
                closed = trajectory_closed(trajectory, z0=z0, closure_tol=args.closure_tol)
                if not closed and not args.allow_open_trajectories:
                    continue

                augmented = generator.add_state_perturbations(
                    trajectory,
                    noise_std=args.noise_std,
                    num_perturbations_per_point=args.dagger_factor,
                ) if args.dagger_factor > 0 else []

                tagged_records = []
                for record in trajectory + augmented:
                    tagged = dict(record)
                    tagged["epsilon"] = float(epsilon)
                    tagged["trajectory_id"] = int(trajectory_id)
                    tagged["matrix_size"] = int(n)
                    tagged["matrix_type"] = matrix_type
                    tagged["point_sampler"] = sampler
                    tagged_records.append(tagged)

                accepted_records = tagged_records
                accepted_meta = {
                    "trajectory_id": int(trajectory_id),
                    "matrix_seed": int(matrix_seed),
                    "matrix_type": matrix_type,
                    "matrix_size": int(n),
                    "sample_mode": args.sample_mode,
                    "point_sampler": sampler,
                    "attempt_index": int(attempt_idx),
                    "epsilon": float(epsilon),
                    "random_point": serialize_complex(z_random),
                    "start_point": serialize_complex(z0),
                    "nearest_eigenvalue": serialize_complex(anchor),
                    "closed": bool(closed),
                    "raw_steps": int(len(trajectory)),
                    "saved_samples": int(len(tagged_records)),
                }
                break

            if accepted_records is None or accepted_meta is None:
                continue

            records.extend(accepted_records)
            trajectory_meta.append(accepted_meta)
            trajectory_id += 1

            stats["num_trajectories"] += 1
            if accepted_meta["closed"]:
                stats["num_closed_trajectories"] += 1
            else:
                stats["num_open_trajectories"] += 1
            stats["matrix_type_counts"][matrix_type] += 1
            stats["matrix_size_counts"][str(n)] += 1
            stats["restart_samples"] += int(sum(int(r["y_restart"]) for r in accepted_records))
            epsilon_values.append(float(accepted_meta["epsilon"]))
            if stats["feature_dim"] is None and accepted_records:
                stats["feature_dim"] = int(np.asarray(accepted_records[0]["features"]).shape[-1])

            print(
                f"[traj {trajectory_id:05d}] type={matrix_type} n={n} "
                f"sampler={accepted_meta['point_sampler']} epsilon={accepted_meta['epsilon']:.4e} "
                f"closed={int(accepted_meta['closed'])} samples={len(accepted_records)} total={len(records)}"
            )

            if len(records) % args.save_every == 0:
                stats["num_samples"] = int(len(records))
                if epsilon_values:
                    stats["epsilon_min"] = float(np.min(epsilon_values))
                    stats["epsilon_max"] = float(np.max(epsilon_values))
                    stats["epsilon_mean"] = float(np.mean(epsilon_values))
                save_dataset(output_dir, f"partial_{len(records)}", records, trajectory_meta, stats, rng)
                print(f"Saved partial dataset with {len(records)} samples to {output_dir}")

    stats["num_samples"] = int(len(records))
    if epsilon_values:
        stats["epsilon_min"] = float(np.min(epsilon_values))
        stats["epsilon_max"] = float(np.max(epsilon_values))
        stats["epsilon_mean"] = float(np.mean(epsilon_values))

    save_dataset(output_dir, "dataset_full", records, trajectory_meta, stats, rng)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
