from __future__ import annotations

import numpy as np

from .contour_init import sigma_min_at


def estimate_tangent_from_sigma_field(
    A: np.ndarray,
    z: complex,
    epsilon: float,
    preferred_direction: complex | None = None,
    relative_step: float = 1.0e-4,
    min_step: float = 1.0e-7,
) -> complex:
    A = np.asarray(A, dtype=np.complex128)
    n = max(int(A.shape[0]), 1)
    matrix_scale = float(np.linalg.norm(A, ord="fro") / max(np.sqrt(float(n)), 1.0))
    base_scale = max(1.0, abs(complex(z)), abs(float(epsilon)), matrix_scale)
    h = max(float(min_step), float(relative_step) * base_scale)

    sigma_x_plus = sigma_min_at(A, z + h)
    sigma_x_minus = sigma_min_at(A, z - h)
    sigma_y_plus = sigma_min_at(A, z + 1j * h)
    sigma_y_minus = sigma_min_at(A, z - 1j * h)

    grad_x = float((sigma_x_plus - sigma_x_minus) / (2.0 * h))
    grad_y = float((sigma_y_plus - sigma_y_minus) / (2.0 * h))
    grad_norm = float(np.hypot(grad_x, grad_y))
    if not np.isfinite(grad_norm) or grad_norm < 1e-12:
        raise ValueError("Local sigma gradient is too small; contour tangent is numerically ill-defined.")

    tangent = complex(-grad_y, grad_x) / grad_norm
    if preferred_direction is not None and abs(preferred_direction) > 1e-12:
        preferred = complex(preferred_direction) / abs(preferred_direction)
        if float(np.real(np.conj(preferred) * tangent)) < 0.0:
            tangent = -tangent
    return tangent
