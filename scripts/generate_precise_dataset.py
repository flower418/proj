from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
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
    if len(records) < 16:
        return False
    if bool(records[-1].get("closed", False)):
        return True

    z_last = complex(records[-1]["z_next"])
    last_ds = max(float(records[-1].get("ds_expert", 0.0)), 1e-8)
    effective_closure_tol = max(float(closure_tol), 0.5 * last_ds)
    if abs(z_last - z0) >= effective_closure_tol:
        return False

    points = np.asarray([z0] + [complex(r["z_next"]) for r in records], dtype=np.complex128)
    path_length = float(np.sum(np.abs(np.diff(points))))
    max_distance_from_start = float(np.max(np.abs(points - z0)))
    min_path_length = max(20.0 * float(closure_tol), 10.0 * last_ds)
    min_escape_distance = max(10.0 * float(closure_tol), 5.0 * last_ds)
    return path_length >= min_path_length and max_distance_from_start >= min_escape_distance


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


def evenly_subsample_records(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("limit must be positive.")
    if len(records) <= limit:
        return records
    stride = len(records) / float(limit)
    indices = [min(int(i * stride), len(records) - 1) for i in range(limit)]
    return [records[idx] for idx in indices]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a precise contour-tracking dataset aligned with the random-point point_sigma task.")
    parser.add_argument("--target-samples", type=int, default=50000)
    parser.add_argument("--target-trajectories", type=int, default=None, help="Optional lower bound on the number of accepted full contours.")
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
    parser.add_argument("--max-samples-per-trajectory", type=int, default=None, help="Optional cap on how many saved samples a single accepted contour can contribute.")
    parser.add_argument("--status-every", type=int, default=20, help="Print a running status summary every this many trajectory attempts.")
    parser.add_argument("--trajectory-heartbeat-every", type=int, default=100, help="Print a heartbeat every this many teacher steps inside one trajectory.")
    parser.add_argument("--max-wall-seconds-per-trajectory", type=float, default=120.0, help="Abort a single teacher trajectory attempt after this many seconds. Set <= 0 to disable.")
    parser.add_argument("--epsilon-min", type=float, default=None, help="Optional lower bound for accepted epsilon.")
    parser.add_argument("--epsilon-max", type=float, default=None, help="Optional upper bound for accepted epsilon.")
    parser.add_argument("--allow-open-trajectories", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wall_start = time.perf_counter()

    if args.target_samples is None and args.target_trajectories is None:
        raise ValueError("At least one of --target-samples or --target-trajectories must be specified.")
    if args.max_samples_per_trajectory is not None and args.max_samples_per_trajectory <= 0:
        raise ValueError("--max-samples-per-trajectory must be positive when provided.")

    rng = np.random.default_rng(args.seed)
    records: list[dict[str, Any]] = []
    trajectory_meta: list[dict[str, Any]] = []
    trajectory_id = 0
    matrix_counter = 0
    total_attempts = 0
    last_save_count = 0
    pairings = [(matrix_type, int(n)) for matrix_type in args.matrix_types for n in args.matrix_sizes]

    stats: dict[str, Any] = {
        "generator": "generate_precise_dataset",
        "sample_mode": args.sample_mode,
        "point_samplers": list(args.point_samplers),
        "target_samples": int(args.target_samples),
        "target_trajectories": None if args.target_trajectories is None else int(args.target_trajectories),
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
        "attempts_total": 0,
        "attempts_projection_failed": 0,
        "attempts_epsilon_filtered": 0,
        "attempts_empty_trajectory": 0,
        "attempts_open_trajectory": 0,
        "attempts_generation_failed": 0,
    }
    epsilon_values: list[float] = []

    def generation_complete() -> bool:
        sample_done = len(records) >= int(args.target_samples)
        if args.target_trajectories is None:
            return sample_done
        trajectory_done = trajectory_id >= int(args.target_trajectories)
        return sample_done and trajectory_done

    def progress_fields() -> dict[str, Any]:
        elapsed_seconds = float(time.perf_counter() - wall_start)
        samples_collected = int(len(records))
        samples_remaining = int(max(args.target_samples - samples_collected, 0))
        progress_pct = float(100.0 * samples_collected / max(int(args.target_samples), 1))
        trajectories_remaining = (
            None
            if args.target_trajectories is None
            else int(max(int(args.target_trajectories) - trajectory_id, 0))
        )
        trajectory_progress_pct = (
            None
            if args.target_trajectories is None
            else float(100.0 * trajectory_id / max(int(args.target_trajectories), 1))
        )
        samples_per_second = float(samples_collected / elapsed_seconds) if elapsed_seconds > 1e-12 else 0.0
        eta_seconds = (
            float(samples_remaining / samples_per_second)
            if samples_per_second > 1e-12
            else None
        )
        avg_samples_per_trajectory = (
            float(samples_collected / trajectory_id)
            if trajectory_id > 0
            else None
        )
        estimated_trajectories_remaining = (
            int(math.ceil(samples_remaining / avg_samples_per_trajectory))
            if avg_samples_per_trajectory is not None and avg_samples_per_trajectory > 1e-12
            else None
        )
        return {
            "target_samples": int(args.target_samples),
            "samples_collected": samples_collected,
            "samples_remaining": samples_remaining,
            "progress_pct": progress_pct,
            "target_trajectories": None if args.target_trajectories is None else int(args.target_trajectories),
            "trajectories_collected": int(trajectory_id),
            "trajectories_remaining": trajectories_remaining,
            "trajectory_progress_pct": trajectory_progress_pct,
            "elapsed_total_seconds": elapsed_seconds,
            "samples_per_second": samples_per_second,
            "eta_seconds": eta_seconds,
            "estimated_trajectories_remaining": estimated_trajectories_remaining,
        }

    print(
        json.dumps(
            {
                "stage": "dataset_generation_started",
                "matrix_sizes": args.matrix_sizes,
                "matrix_types": args.matrix_types,
                "sample_mode": args.sample_mode,
                "point_samplers": args.point_samplers,
                "max_steps": args.max_steps,
                "dagger_factor": args.dagger_factor,
                "output_dir": str(output_dir),
                "status_every": args.status_every,
                **progress_fields(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    while not generation_complete():
        matrix_type, n = pairings[matrix_counter % len(pairings)]
        matrix_seed = args.seed + matrix_counter
        matrix_counter += 1

        A = build_matrix(matrix_type=matrix_type, n=n, seed=matrix_seed).astype(np.complex128)
        solver = PseudoinverseSolver(method="minres", tol=args.solver_tol, max_iter=1000)

        print(
            json.dumps(
                {
                    "stage": "matrix_started",
                    "matrix_index": matrix_counter,
                    "matrix_seed": matrix_seed,
                    "matrix_type": matrix_type,
                    "matrix_size": n,
                    **progress_fields(),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        for local_traj_idx in range(args.trajectories_per_type):
            if generation_complete():
                break

            accepted_records = None
            accepted_meta = None
            for attempt_idx in range(args.max_attempts_per_trajectory):
                total_attempts += 1
                stats["attempts_total"] = int(total_attempts)
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
                        stats["attempts_projection_failed"] += 1
                        continue
                    epsilon = float(sigma_min_at(A, z0))
                else:
                    z0 = complex(z_random)
                    epsilon = float(sigma_min_at(A, z0))

                if args.epsilon_min is not None and epsilon < args.epsilon_min:
                    stats["attempts_epsilon_filtered"] += 1
                    continue
                if args.epsilon_max is not None and epsilon > args.epsilon_max:
                    stats["attempts_epsilon_filtered"] += 1
                    continue

                print(
                    json.dumps(
                        {
                            "stage": "attempt_started",
                            "attempts_total": total_attempts,
                            "trajectory_slot": trajectory_id,
                            "matrix_type": matrix_type,
                            "matrix_size": n,
                            "point_sampler": sampler,
                            "epsilon": float(epsilon),
                            **progress_fields(),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

                def _trajectory_step_callback(record: dict[str, Any]) -> None:
                    heartbeat_every = max(int(args.trajectory_heartbeat_every), 1)
                    step_idx = int(record["step"])
                    if step_idx == 0 or ((step_idx + 1) % heartbeat_every == 0):
                        print(
                            json.dumps(
                                {
                                    "stage": "trajectory_step",
                                    "attempts_total": total_attempts,
                                    "matrix_type": matrix_type,
                                    "matrix_size": n,
                                    "step": step_idx,
                                    "elapsed_seconds": float(record.get("elapsed_seconds", 0.0)),
                                    "ds_expert": float(record["ds_expert"]),
                                    "y_restart": int(record["y_restart"]),
                                    "applied_projection": int(record.get("applied_projection", False)),
                                    "backtracks": int(record.get("backtracks", 0)),
                                    "sigma_error": float(record["sigma_error"]),
                                    "residual": float(record["residual"]),
                                    **progress_fields(),
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )

                try:
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
                    trajectory = generator.generate_trajectory(
                        z0=z0,
                        max_steps=args.max_steps,
                        step_callback=_trajectory_step_callback,
                        max_wall_seconds=(
                            None
                            if args.max_wall_seconds_per_trajectory is None or args.max_wall_seconds_per_trajectory <= 0
                            else float(args.max_wall_seconds_per_trajectory)
                        ),
                    )
                except Exception:
                    stats["attempts_generation_failed"] += 1
                    continue
                if not trajectory:
                    stats["attempts_empty_trajectory"] += 1
                    continue
                if (
                    args.max_wall_seconds_per_trajectory is not None
                    and args.max_wall_seconds_per_trajectory > 0
                    and float(trajectory[-1].get("elapsed_seconds", 0.0)) >= float(args.max_wall_seconds_per_trajectory)
                ):
                    stats["attempts_generation_failed"] += 1
                    print(
                        json.dumps(
                            {
                                "stage": "trajectory_timeout",
                                "attempts_total": total_attempts,
                                "matrix_type": matrix_type,
                                "matrix_size": n,
                                "elapsed_seconds": float(trajectory[-1].get("elapsed_seconds", 0.0)),
                                "steps_completed": int(len(trajectory)),
                                **progress_fields(),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    continue
                closed = trajectory_closed(trajectory, z0=z0, closure_tol=args.closure_tol)
                if not closed and not args.allow_open_trajectories:
                    stats["attempts_open_trajectory"] += 1
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
                raw_record_count = int(len(trajectory))
                augmented_record_count = int(len(augmented))
                if args.max_samples_per_trajectory is not None:
                    tagged_records = evenly_subsample_records(tagged_records, int(args.max_samples_per_trajectory))

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
                    "raw_saved_samples": raw_record_count,
                    "augmented_saved_samples": augmented_record_count,
                }
                break

            if accepted_records is None or accepted_meta is None:
                print(
                    json.dumps(
                        {
                            "stage": "attempt_rejected",
                            "attempts_total": total_attempts,
                            "matrix_type": matrix_type,
                            "matrix_size": n,
                            "reasons": {
                                "projection_failed": stats["attempts_projection_failed"],
                                "epsilon_filtered": stats["attempts_epsilon_filtered"],
                                "empty_trajectory": stats["attempts_empty_trajectory"],
                                "open_trajectory": stats["attempts_open_trajectory"],
                                "generation_failed": stats["attempts_generation_failed"],
                            },
                            **progress_fields(),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if args.status_every > 0 and total_attempts % args.status_every == 0:
                    print(
                        json.dumps(
                            {
                                "stage": "status",
                                "attempts_total": total_attempts,
                                "projection_failed": stats["attempts_projection_failed"],
                                "epsilon_filtered": stats["attempts_epsilon_filtered"],
                                "empty_trajectory": stats["attempts_empty_trajectory"],
                                "open_trajectory": stats["attempts_open_trajectory"],
                                "generation_failed": stats["attempts_generation_failed"],
                                **progress_fields(),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
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
                json.dumps(
                    {
                        "stage": "trajectory_accepted",
                        "trajectory_id": trajectory_id,
                        "matrix_type": matrix_type,
                        "matrix_size": n,
                        "point_sampler": accepted_meta["point_sampler"],
                        "epsilon": accepted_meta["epsilon"],
                        "closed": accepted_meta["closed"],
                        "saved_samples": len(accepted_records),
                        "attempts_total": total_attempts,
                        **progress_fields(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

            if args.status_every > 0 and total_attempts % args.status_every == 0:
                print(
                    json.dumps(
                    {
                        "stage": "status",
                        "attempts_total": total_attempts,
                        "projection_failed": stats["attempts_projection_failed"],
                        "epsilon_filtered": stats["attempts_epsilon_filtered"],
                        "empty_trajectory": stats["attempts_empty_trajectory"],
                        "open_trajectory": stats["attempts_open_trajectory"],
                        "generation_failed": stats["attempts_generation_failed"],
                        **progress_fields(),
                    },
                    ensure_ascii=False,
                        ),
                        flush=True,
                    )

            if len(records) - last_save_count >= args.save_every:
                stats["num_samples"] = int(len(records))
                if epsilon_values:
                    stats["epsilon_min"] = float(np.min(epsilon_values))
                    stats["epsilon_max"] = float(np.max(epsilon_values))
                    stats["epsilon_mean"] = float(np.mean(epsilon_values))
                save_dataset(output_dir, f"partial_{len(records)}", records, trajectory_meta, stats, rng)
                last_save_count = len(records)
                print(
                    json.dumps(
                        {
                            "stage": "partial_saved",
                            "output_dir": str(output_dir),
                            **progress_fields(),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    stats["num_samples"] = int(len(records))
    if epsilon_values:
        stats["epsilon_min"] = float(np.min(epsilon_values))
        stats["epsilon_max"] = float(np.max(epsilon_values))
        stats["epsilon_mean"] = float(np.mean(epsilon_values))

    save_dataset(output_dir, "dataset_full", records, trajectory_meta, stats, rng)
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
