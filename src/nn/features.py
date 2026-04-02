from __future__ import annotations

import numpy as np


def _log_normalize(value: float, min_exp: float = -12.0, max_exp: float = 0.0) -> float:
    value = max(float(value), 10 ** min_exp)
    log_value = np.log10(value)
    scaled = (log_value - min_exp) / (max_exp - min_exp)
    return float(np.clip(2.0 * scaled - 1.0, -1.0, 1.0))


def extract_features(
    z: complex,
    u: np.ndarray,
    v: np.ndarray,
    A: np.ndarray,
    epsilon: float,
    prev_gamma_arg: float | None = None,
    prev_solver_iters: int = 0,
    max_iter_scale: int = 500,
    normalize: bool = True,
) -> np.ndarray:
    n = len(u)
    M = z * np.eye(n, dtype=np.complex128) - np.asarray(A, dtype=np.complex128)
    sigma_approx = np.abs(np.vdot(u, M @ v))
    f1 = np.abs(sigma_approx - epsilon)
    f2 = np.abs(1.0 - np.linalg.norm(u))
    f3 = np.abs(1.0 - np.linalg.norm(v))
    residual = M @ v - epsilon * u
    f4 = np.linalg.norm(residual)
    gamma = np.vdot(u, v)
    f5 = np.abs(gamma)

    if prev_gamma_arg is None:
        f6 = 0.0
    else:
        delta = np.angle(gamma) - prev_gamma_arg
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        f6 = np.abs(delta)

    f7 = float(prev_solver_iters)
    matrix_scale = np.linalg.norm(np.asarray(A, dtype=np.complex128), ord="fro") / max(np.sqrt(float(n)), 1.0)
    f8 = np.abs(float(epsilon))
    f9 = float(matrix_scale)
    f10 = np.abs(z) / max(float(matrix_scale), 1e-12)
    if normalize:
        features = np.array(
            [
                _log_normalize(f1),
                _log_normalize(f2),
                _log_normalize(f3),
                _log_normalize(f4),
                _log_normalize(f5, min_exp=-6.0, max_exp=0.0),
                float(np.clip(f6 / np.pi, 0.0, 1.0)),
                float(np.clip(f7 / max(max_iter_scale, 1), 0.0, 1.0)),
                _log_normalize(f8, min_exp=-12.0, max_exp=2.0),
                _log_normalize(f9, min_exp=-6.0, max_exp=3.0),
                _log_normalize(f10, min_exp=-6.0, max_exp=3.0),
            ],
            dtype=np.float32,
        )
        return features
    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10], dtype=np.float32)
