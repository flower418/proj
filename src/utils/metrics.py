from __future__ import annotations

import numpy as np


def residual_norm(A: np.ndarray, z: complex, u: np.ndarray, v: np.ndarray, epsilon: float) -> float:
    M = z * np.eye(A.shape[0], dtype=np.complex128) - A
    return float(np.linalg.norm(M @ v - epsilon * u))


def contour_closure_error(trajectory: np.ndarray) -> float:
    if len(trajectory) < 2:
        return 0.0
    return float(np.abs(trajectory[-1] - trajectory[0]))
