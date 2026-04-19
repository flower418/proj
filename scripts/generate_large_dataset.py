from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import _bootstrap  # noqa: F401

from src.core.contour_tracker import ContourTracker
from src.nn.features import assemble_controller_features, extract_features
from src.train.expert_solver import ExpertSolver
from src.utils.demo_sampling import (
    SUPPORTED_MATRIX_TYPES,
    build_random_matrix,
    sample_training_contour_start,
)
from src.utils.run_logging import RunLogger


PROGRESS_LOG_EVERY = 20
MIN_SAMPLES_PER_TRAJECTORY = 24


def generate_trajectory(
    A: np.ndarray,
    epsilon: float,
    z0: complex,
    max_steps: int = 96,
    step_progress_callback=None,
) -> List[Dict]:
    """生成单条更接近推理分布的 teacher-forced 轨迹。"""
    expert = ExpertSolver(A=A, epsilon=epsilon)
    tracker = ContourTracker(
        A=A,
        epsilon=epsilon,
        controller=None,
        fixed_step_size=expert.first_step,
        projection_tol=expert.projection_tol,
        min_step_size=expert.min_step_size,
    )

    u, v = tracker.initialize(z0)
    z = complex(z0)
    step_hint = expert.first_step
    prev_ds = 0.0
    trajectory = []
    path_length = 0.0
    max_distance_from_start = 0.0
    closure_anchor = tracker._closure_anchor(z0)
    prev_anchor_angle = None if abs(z0 - closure_anchor) < 1e-12 else float(np.angle(z0 - closure_anchor))
    winding_angle = 0.0

    for step_idx in range(max_steps):
        z_prev = z
        expert_result = expert.step_with_hint(z, u, v, step_hint=step_hint)

        base_features = extract_features(
            z=z,
            u=u,
            v=v,
            A=A,
            epsilon=epsilon,
        )
        features = assemble_controller_features(
            base_features,
            prev_ds=prev_ds,
        )

        z_next, u_next, v_next, _, step_info = tracker._advance_step(
            z=z,
            u=u,
            v=v,
            ds=max(float(expert_result.ds_expert), expert.min_step_size),
        )
        applied_projection = bool(step_info["applied_projection"])
        step_distance = float(np.abs(z_next - z))
        path_length += step_distance
        max_distance_from_start = max(max_distance_from_start, float(np.abs(z_next - z0)))
        if prev_anchor_angle is not None and abs(z_next - closure_anchor) >= 1e-12:
            current_anchor_angle = float(np.angle(z_next - closure_anchor))
            delta = current_anchor_angle - prev_anchor_angle
            delta = float(np.angle(np.exp(1j * delta)))
            winding_angle += delta
            prev_anchor_angle = current_anchor_angle
        elif abs(z_next - closure_anchor) >= 1e-12:
            prev_anchor_angle = float(np.angle(z_next - closure_anchor))

        trajectory.append(
            {
                "features": features,
                "ds_expert": float(expert_result.ds_expert),
            }
        )
        if step_progress_callback is not None and (step_idx + 1) % PROGRESS_LOG_EVERY == 0:
            step_progress_callback(
                {
                    "step": int(step_idx + 1),
                    "max_steps": int(max_steps),
                    "z": z_next,
                    "step_distance": float(step_distance),
                    "path_length": float(path_length),
                    "winding_angle": float(winding_angle),
                    "applied_projection": bool(applied_projection),
                }
            )

        z, u, v = z_next, u_next, v_next
        prev_ds = float(expert_result.ds_expert)
        step_hint = float(expert_result.suggested_next_step)

        if tracker.check_closure(
            z_current=z,
            z_start=z0,
            current_step=step_idx + 1,
            path_length=path_length,
            max_distance_from_start=max_distance_from_start,
            winding_angle=winding_angle,
            last_step_size=max(float(expert_result.ds_expert), tracker.min_step_size),
            z_prev=z_prev,
        ):
            break

    return trajectory


def _log_message(run_logger: RunLogger | None, message: str) -> None:
    if run_logger is None:
        print(message)
    else:
        run_logger.log(message)


def _make_progress_printer(run_logger: RunLogger | None):
    def _printer(info: dict) -> None:
        step = int(info["step"])
        max_steps = int(info.get("max_steps", 0))
        _log_message(run_logger, f"      Teacher {step}/{max_steps}")

    return _printer


def _extend_records_with_metadata(
    all_records: List[Dict],
    records: List[Dict],
    *,
    trajectory_id: str,
) -> int:
    for record in records:
        enriched = dict(record)
        enriched["trajectory_id"] = trajectory_id
        all_records.append(enriched)
    return len(records)


def generate_diverse_dataset(
    target_samples: int = 10000,
    matrix_sizes: List[int] = None,
    trajectories_per_type: int = 5,
    max_steps: int = 96,
    near_eigen_ratio: float = 0.75,
    output_dir: str = "data",
    save_every: int = 5000,
    seed: int = 0,
    run_logger: RunLogger | None = None,
) -> Dict:
    """生成多样化数据集"""
    if matrix_sizes is None:
        matrix_sizes = [30, 50, 80]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_records = []
    stats = {
        "total_samples": 0,
        "matrix_types": {},
        "matrix_sizes": {},
        "trajectories": 0,
        "supported_matrix_types": list(SUPPORTED_MATRIX_TYPES),
        "sampling_strategy": "mixed_near_eigen_and_random_point",
        "near_eigen_ratio": float(near_eigen_ratio),
        "feature_dim": None,
        "sampling_modes": {"near_eigen": 0, "random_point": 0},
    }
    rng = np.random.default_rng(seed)
    matrix_counter = 0
    size_cursor = 0
    last_saved_count = 0
    while len(all_records) < target_samples:
        n = int(matrix_sizes[size_cursor % len(matrix_sizes)])
        size_cursor += 1

        for traj_idx in range(trajectories_per_type):
            if len(all_records) >= target_samples:
                break

            matrix_type, A = build_random_matrix(n=n, rng=rng)
            matrix_id = f"{matrix_type}_n{n}_m{matrix_counter:06d}"
            matrix_counter += 1
            _log_message(run_logger, f"生成：{matrix_type}, n={n}")

            target_scale = max(len(matrix_sizes) * len(SUPPORTED_MATRIX_TYPES) * trajectories_per_type * 50, 1)
            num_starts = max(2, min(8, target_samples // target_scale))

            for start_idx in range(num_starts):
                try:
                    z0, epsilon, z_random, anchor, sample_meta = sample_training_contour_start(
                        A=A,
                        rng=rng,
                        near_eigen_ratio=near_eigen_ratio,
                    )
                except Exception as e:
                    _log_message(run_logger, f"    起点采样失败：{e}")
                    continue

                sampling_mode = str(sample_meta.get("sampling_mode", "unknown"))
                sampling_radius = sample_meta.get("sampling_radius")
                _log_message(
                    run_logger,
                    f"  轨迹 {traj_idx + 1}, 起点 {start_idx + 1}: "
                    f"mode={sampling_mode}, epsilon={epsilon:.6f}, z0 = {z0:.3f}"
                )

                try:
                    step_progress_printer = _make_progress_printer(run_logger)
                    trajectory = generate_trajectory(
                        A,
                        epsilon,
                        z0,
                        max_steps,
                        step_progress_callback=step_progress_printer,
                    )
                    if len(trajectory) < MIN_SAMPLES_PER_TRAJECTORY:
                        _log_message(run_logger, f"    轨迹过短：{len(trajectory)} 步，跳过")
                        continue
                    _log_message(
                        run_logger,
                        f"    Teacher 完成：{len(trajectory)} 步"
                    )
                    trajectory_id = f"{matrix_id}_s{start_idx:03d}"
                    raw_count = _extend_records_with_metadata(
                        all_records,
                        trajectory,
                        trajectory_id=trajectory_id,
                    )
                    stats["total_samples"] += raw_count

                    if stats["feature_dim"] is None and trajectory:
                        stats["feature_dim"] = int(np.asarray(trajectory[0]["features"]).shape[-1])

                    stats["trajectories"] += 1
                    stats["sampling_modes"][sampling_mode] = int(stats["sampling_modes"].get(sampling_mode, 0) + 1)
                    total_added = raw_count
                    stats["matrix_types"][matrix_type] = int(stats["matrix_types"].get(matrix_type, 0) + total_added)
                    size_key = str(n)
                    stats["matrix_sizes"][size_key] = int(stats["matrix_sizes"].get(size_key, 0) + total_added)
                    if run_logger is not None:
                        run_logger.append_jsonl(
                            "progress.jsonl",
                            {
                                "event": "trajectory_done",
                                "matrix_type": matrix_type,
                                "matrix_size": n,
                                "trajectory_index": traj_idx,
                                "start_index": start_idx,
                                "epsilon": float(epsilon),
                                "random_point": [float(np.real(z_random)), float(np.imag(z_random))],
                                "start_point": [float(np.real(z0)), float(np.imag(z0))],
                                "nearest_eigenvalue": [float(np.real(anchor)), float(np.imag(anchor))],
                                "sampling_mode": sampling_mode,
                                "sampling_radius": sampling_radius,
                                "raw_samples": raw_count,
                                "total_samples": stats["total_samples"],
                            },
                        )

                    if save_every > 0 and len(all_records) - last_saved_count >= save_every:
                        save_dataset(all_records, output_path, f"partial_{len(all_records)}", rng=rng)
                        last_saved_count = len(all_records)
                        _log_message(run_logger, f"    已保存 {len(all_records)} 样本")
                        if run_logger is not None:
                            run_logger.append_jsonl(
                                "progress.jsonl",
                                {
                                    "event": "partial_save",
                                    "total_samples": len(all_records),
                                    "save_every": save_every,
                                },
                            )

                    if len(all_records) >= target_samples:
                        break

                except Exception as e:
                    _log_message(run_logger, f"    失败：{e}")
                    if run_logger is not None:
                        run_logger.append_jsonl(
                            "progress.jsonl",
                            {
                                "event": "trajectory_failed",
                                "matrix_type": matrix_type,
                                "matrix_size": n,
                                "trajectory_index": traj_idx,
                                "start_index": start_idx,
                                "error": str(e),
                            },
                        )
                    continue

            if len(all_records) >= target_samples:
                break

    save_dataset(all_records, output_path, "dataset_full", stats, rng=rng)
    return stats


def save_dataset(records: List[Dict], output_path: Path, name: str, stats: dict = None, rng: np.random.Generator | None = None):
    """保存数据集"""
    features = np.stack([r["features"] for r in records])
    ds_expert = np.array([r["ds_expert"] for r in records], dtype=np.float32)
    trajectory_id = np.array([str(r.get("trajectory_id", f"trajectory_{idx}")) for idx, r in enumerate(records)], dtype=np.str_)

    np.savez(
        output_path / f"{name}.npz",
        features=features,
        ds_expert=ds_expert,
    )

    n = len(records)
    rng = rng or np.random.default_rng(0)
    unique_trajectories = np.asarray(np.unique(trajectory_id), dtype=np.str_)
    if len(unique_trajectories) < 3:
        indices = rng.permutation(n)
        train_end, val_end = int(0.8 * n), int(0.9 * n)
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
    else:
        rng.shuffle(unique_trajectories)
        num_groups = len(unique_trajectories)
        train_group_end = max(1, int(0.8 * num_groups))
        val_group_end = max(train_group_end + 1, int(0.9 * num_groups))
        train_groups = set(unique_trajectories[:train_group_end].tolist())
        val_groups = set(unique_trajectories[train_group_end:val_group_end].tolist())
        test_groups = set(unique_trajectories[val_group_end:].tolist())
        train_indices = np.nonzero(np.isin(trajectory_id, list(train_groups)))[0]
        val_indices = np.nonzero(np.isin(trajectory_id, list(val_groups)))[0]
        test_indices = np.nonzero(np.isin(trajectory_id, list(test_groups)))[0]
    split_payload = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }

    np.savez(output_path / f"{name}_splits.npz", **split_payload)
    if name == "dataset_full":
        np.savez(output_path / "dataset_splits.npz", **split_payload)

    if stats:
        with open(output_path / "dataset_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\n保存：{n} 样本")
        print(f"  Train: {len(train_indices)} ({100*len(train_indices)/n:.1f}%)")
        print(f"  Val: {len(val_indices)} ({100*len(val_indices)/n:.1f}%)")
        print(f"  Test: {len(test_indices)} ({100*len(test_indices)/n:.1f}%)")


def parse_args():
    parser = argparse.ArgumentParser(description="生成大规模数据集")
    parser.add_argument("--target-samples", type=int, default=10000)
    parser.add_argument("--matrix-sizes", type=int, nargs="+", default=[30, 50, 80])
    parser.add_argument("--trajectories-per-type", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--near-eigen-ratio", type=float, default=0.75)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default=None, help="Directory for runtime logs. Defaults to <output-dir>/logs.")
    return parser.parse_args()


def main():
    args = parse_args()
    log_root = Path(args.log_dir) if args.log_dir is not None else Path(args.output_dir) / "logs"
    with RunLogger(log_root, run_name="generate_large_dataset") as run_logger:
        run_logger.write_json("run_config.json", {"args": vars(args)})
        stats = generate_diverse_dataset(
            target_samples=args.target_samples,
            matrix_sizes=args.matrix_sizes,
            trajectories_per_type=args.trajectories_per_type,
            max_steps=args.max_steps,
            near_eigen_ratio=args.near_eigen_ratio,
            output_dir=args.output_dir,
            save_every=args.save_every,
            seed=args.seed,
            run_logger=run_logger,
        )
        summary_path = run_logger.write_json("generation_summary.json", stats)
        run_logger.log(
            f"完成！总样本：{stats['total_samples']:,}, "
            f"log_dir={run_logger.log_dir}, summary={summary_path}"
        )


if __name__ == "__main__":
    main()
