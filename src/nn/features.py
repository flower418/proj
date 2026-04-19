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
    normalize: bool = True,
) -> np.ndarray:
    n = len(u)
    M = z * np.eye(n, dtype=np.complex128) - np.asarray(A, dtype=np.complex128)
    sigma_approx = np.abs(np.vdot(u, M @ v))
    sigma_error = np.abs(sigma_approx - epsilon)
    triplet_residual = np.linalg.norm(M @ v - epsilon * u)
    uv_overlap = np.abs(np.vdot(u, v))
    epsilon_abs = np.abs(float(epsilon))
    matrix_scale = np.linalg.norm(np.asarray(A, dtype=np.complex128), ord="fro") / max(np.sqrt(float(n)), 1.0)
    if normalize:
        features = np.array(
            [
                _log_normalize(sigma_error),
                _log_normalize(triplet_residual),
                _log_normalize(uv_overlap, min_exp=-6.0, max_exp=0.0),
                _log_normalize(epsilon_abs, min_exp=-12.0, max_exp=2.0),
                _log_normalize(matrix_scale, min_exp=-6.0, max_exp=3.0),
            ],
            dtype=np.float32,
        )
        return features
    return np.array([sigma_error, triplet_residual, uv_overlap, epsilon_abs, matrix_scale], dtype=np.float32)


def assemble_controller_features(
    base_features: np.ndarray,
    prev_ds: float = 0.0,
    input_dim: int | None = None,
) -> np.ndarray:
    base = np.asarray(base_features, dtype=np.float32).reshape(-1)
    context = np.array(
        [
            _log_normalize(float(prev_ds), min_exp=-8.0, max_exp=-1.0),
        ],
        dtype=np.float32,
    )
    full = np.concatenate([base, context], axis=0)
    if input_dim is None:
        return full
    if input_dim != len(full):
        raise ValueError(
            f"Feature dimension mismatch: assembled {len(full)} features, but controller expects {input_dim}."
        )
    return full
