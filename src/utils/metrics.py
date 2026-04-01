from __future__ import annotations

import numpy as np


def residual_norm(A: np.ndarray, z: complex, u: np.ndarray, v: np.ndarray, epsilon: float) -> float:
    M = z * np.eye(A.shape[0], dtype=np.complex128) - A
    return float(np.linalg.norm(M @ v - epsilon * u))


def contour_closure_error(trajectory: np.ndarray) -> float:
    if len(trajectory) < 2:
        return 0.0
    return float(np.abs(trajectory[-1] - trajectory[0]))


def step_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if y_true.size == 0:
        return {
            "step_size_mae": 0.0,
            "step_size_rmse": 0.0,
            "step_size_r2": 0.0,
        }

    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))

    centered = y_true - float(np.mean(y_true))
    ss_tot = float(np.sum(centered**2))
    ss_res = float(np.sum(error**2))
    if ss_tot <= 1e-15:
        r2 = 1.0 if ss_res <= 1e-15 else 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot

    return {
        "step_size_mae": mae,
        "step_size_rmse": rmse,
        "step_size_r2": float(r2),
    }
