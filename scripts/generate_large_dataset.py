from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.optimize import brentq

from src.core.pseudoinverse import PseudoinverseSolver
from src.nn.features import extract_features
from src.train.expert_solver import ExpertSolver
from src.train.dagger_augmentation import DAggerAugmenter


class MatrixGenerator:
    """生成各种类型的矩阵"""

    @staticmethod
    def random_complex(n: int, seed: int = 0) -> np.ndarray:
        """随机稠密复矩阵"""
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))

    @staticmethod
    def random_hermitian(n: int, seed: int = 0) -> np.ndarray:
        """随机 Hermitian 矩阵"""
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        return (A + A.conj().T) / 2

    @staticmethod
    def random_real(n: int, seed: int = 0) -> np.ndarray:
        """随机实矩阵"""
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n, n))

    @staticmethod
    def ill_conditioned(n: int, condition_number: float = 1e6, seed: int = 0) -> np.ndarray:
        """病态矩阵"""
        rng = np.random.default_rng(seed)
        U, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
        V, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
        sigma = np.geomspace(1.0, 1.0 / condition_number, n)
        return U @ np.diag(sigma) @ V.conj().T


def find_contour_point(A: np.ndarray, epsilon: float, angle: float, max_iter: int = 50):
    """在指定角度找到伪谱等高线上的点"""
    n = A.shape[0]

    def objective(r):
        z = r * np.exp(1j * angle)
        M = z * np.eye(n, dtype=np.complex128) - A
        return np.min(np.linalg.svd(M, compute_uv=False)) - epsilon

    try:
        r_sol = brentq(objective, 0.1, 5.0, maxiter=max_iter)
        return r_sol * np.exp(1j * angle)
    except ValueError:
        return None


def generate_trajectory(
    A: np.ndarray,
    epsilon: float,
    z0: complex,
    max_steps: int = 200,
    solver_config: dict = None,
) -> List[Dict]:
    """生成单条轨迹"""
    solver = PseudoinverseSolver(**(solver_config or {}))
    expert = ExpertSolver(A=A, epsilon=epsilon, solver=solver)

    _, u, v = expert.svd_solver(A, z0)
    u = u / max(np.linalg.norm(u), 1e-15)
    v = v / max(np.linalg.norm(v), 1e-15)

    z = z0
    steps_since_restart = 0
    step_hint = expert.first_step
    trajectory = []

    for step_idx in range(max_steps):
        result = expert._step_with_hint(z, u, v, steps_since_restart, step_hint)

        prev_gamma_arg = None if step_idx == 0 else float(np.angle(np.vdot(trajectory[-1]["u"], trajectory[-1]["v"])))
        features = extract_features(
            z=z, u=u, v=v, A=A, epsilon=epsilon,
            prev_gamma_arg=prev_gamma_arg,
            prev_solver_iters=expert.ode.solver.get_iteration_count(),
        )

        trajectory.append({
            "z": z,
            "u": u.copy(),
            "v": v.copy(),
            "features": features,
            "ds_expert": result.ds_expert,
            "y_restart": result.y_restart,
            "residual": result.residual,
            "sigma_error": result.sigma_error,
        })

        z, u, v = result.z_next, result.u_next, result.v_next
        steps_since_restart = 0 if result.y_restart else steps_since_restart + 1
        step_hint = max(min(result.ds_expert, expert.max_step), 1e-8)

        if step_idx >= 10 and abs(z - z0) < expert.closure_tol:
            break

    return trajectory


def augment_trajectory(trajectory: List[Dict], A: np.ndarray, epsilon: float, num_perturbations: int = 2):
    """DAgger 增强"""
    solver = PseudoinverseSolver()
    expert = ExpertSolver(A=A, epsilon=epsilon, solver=solver)
    augmenter = DAggerAugmenter(expert)
    return augmenter.augment_trajectory(trajectory, num_perturbations_per_point=num_perturbations)


def generate_diverse_dataset(
    target_samples: int = 10000,
    matrix_sizes: List[int] = None,
    matrix_types: List[str] = None,
    trajectories_per_type: int = 5,
    max_steps: int = 200,
    dagger_factor: int = 2,
    output_dir: str = "data",
    save_every: int = 5000,
    seed: int = 0,
) -> Dict:
    """生成多样化数据集"""
    if matrix_sizes is None:
        matrix_sizes = [30, 50, 80]
    if matrix_types is None:
        matrix_types = ["random_complex", "random_hermitian", "random_real", "ill_conditioned"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_records = []
    stats = {"total_samples": 0, "matrix_types": {}, "matrix_sizes": {}, "trajectories": 0, "restart_samples": 0}
    rng = np.random.default_rng(seed)

    for matrix_type in matrix_types:
        for n in matrix_sizes:
            if len(all_records) >= target_samples:
                break

            print(f"\n生成：{matrix_type}, n={n}")

            for traj_idx in range(trajectories_per_type):
                if len(all_records) >= target_samples:
                    break

                # 生成矩阵
                A = getattr(MatrixGenerator, matrix_type)(n, seed=len(all_records))

                # 生成多个起点
                num_starts = max(2, target_samples // (len(matrix_types) * len(matrix_sizes) * trajectories_per_type * 50))

                for start_idx in range(num_starts):
                    angle = 2 * np.pi * start_idx / num_starts + rng.uniform(-0.2, 0.2)
                    z0 = find_contour_point(A, 0.1, angle)

                    if z0 is None:
                        continue

                    print(f"  轨迹 {traj_idx + 1}, 起点 {start_idx + 1}: z0 = {z0:.3f}")

                    try:
                        trajectory = generate_trajectory(A, 0.1, z0, max_steps)

                        # 添加原始记录
                        for record in trajectory:
                            record["matrix_type"] = matrix_type
                            record["matrix_size"] = n
                            all_records.append(record)
                            stats["total_samples"] += 1
                            if record["y_restart"] == 1:
                                stats["restart_samples"] += 1

                        # DAgger 增强
                        if dagger_factor > 0:
                            augmented = augment_trajectory(trajectory, A, 0.1, dagger_factor)
                            for record in augmented:
                                record["matrix_type"] = matrix_type
                                record["matrix_size"] = n
                                record["source"] = "dagger"
                                all_records.append(record)
                                stats["total_samples"] += 1
                                if record["y_restart"] == 1:
                                    stats["restart_samples"] += 1

                        stats["trajectories"] += 1

                        # 定期保存
                        if len(all_records) % save_every == 0:
                            save_dataset(all_records, output_path, f"partial_{len(all_records)}")
                            print(f"    已保存 {len(all_records)} 样本")

                    except Exception as e:
                        print(f"    失败：{e}")
                        continue

    # 最终保存
    save_dataset(all_records, output_path, "dataset_full", stats)
    return stats


def save_dataset(records: List[Dict], output_path: Path, name: str, stats: dict = None):
    """保存数据集"""
    features = np.stack([r["features"] for r in records])
    ds_expert = np.array([r["ds_expert"] for r in records], dtype=np.float32)
    y_restart = np.array([r["y_restart"] for r in records], dtype=np.int64)

    np.savez(output_path / f"{name}.npz", features=features, ds_expert=ds_expert, y_restart=y_restart)

    # 保存划分
    n = len(records)
    indices = np.random.permutation(n)
    train_end, val_end = int(0.8 * n), int(0.9 * n)

    np.savez(
        output_path / f"{name}_splits.npz",
        train_indices=indices[:train_end],
        val_indices=indices[train_end:val_end],
        test_indices=indices[val_end:],
    )

    if stats:
        with open(output_path / "dataset_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\n保存：{n} 样本")
        print(f"  Train: {train_end} ({100*train_end/n:.1f}%)")
        print(f"  Val: {val_end - train_end} ({100*(val_end-train_end)/n:.1f}%)")
        print(f"  Test: {n - val_end} ({100*(n-val_end)/n:.1f}%)")
        print(f"  重启样本：{stats['restart_samples']} ({100*stats['restart_samples']/n:.2f}%)")


def parse_args():
    parser = argparse.ArgumentParser(description="生成大规模数据集")
    parser.add_argument("--target-samples", type=int, default=10000)
    parser.add_argument("--matrix-sizes", type=int, nargs="+", default=[30, 50, 80])
    parser.add_argument("--matrix-types", type=str, nargs="+", default=None)
    parser.add_argument("--trajectories-per-type", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--dagger-factor", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    stats = generate_diverse_dataset(
        target_samples=args.target_samples,
        matrix_sizes=args.matrix_sizes,
        matrix_types=args.matrix_types,
        trajectories_per_type=args.trajectories_per_type,
        max_steps=args.max_steps,
        dagger_factor=args.dagger_factor,
        output_dir=args.output_dir,
        save_every=args.save_every,
        seed=args.seed,
    )

    print(f"\n完成！总样本：{stats['total_samples']:,}, 重启：{stats['restart_samples']:,}")


if __name__ == "__main__":
    main()
