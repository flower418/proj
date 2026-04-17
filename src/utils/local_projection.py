from __future__ import annotations

from typing import Callable

import numpy as np


def project_to_contour_by_local_normal(
    A: np.ndarray,
    epsilon: float,
    z_candidate: complex,
    svd_solver: Callable[[np.ndarray, complex], tuple[float, np.ndarray, np.ndarray]],
    projection_tol: float,
    max_newton_iters: int = 2,
    sigma_current: float | None = None,
    u_current: np.ndarray | None = None,
    v_current: np.ndarray | None = None,
):
    """Cheap local normal correction without step halving.

    We reuse the local singular-triplet geometry and take a few direct Newton-like
    corrections along the estimated normal direction. If the correction cannot be
    improved quickly, the caller can fall back to a more expensive global projection.
    """

    tol = max(float(projection_tol), 1e-10)
    if sigma_current is None or u_current is None or v_current is None:
        sigma_current, u_current, v_current = svd_solver(A, z_candidate)

    sigma_current = float(sigma_current)
    u_current = np.asarray(u_current, dtype=np.complex128)
    v_current = np.asarray(v_current, dtype=np.complex128)
    u_current = u_current / max(np.linalg.norm(u_current), 1e-15)
    v_current = v_current / max(np.linalg.norm(v_current), 1e-15)

    current_error = abs(sigma_current - float(epsilon))
    if current_error <= tol:
        return z_candidate, u_current, v_current, {
            'sigma': float(sigma_current),
            'sigma_error': float(current_error),
            'projection_distance': 0.0,
            'projection_iterations': 0,
            'projection_mode': 'local_exact',
        }

    z_current = complex(z_candidate)
    for iteration in range(1, max(int(max_newton_iters), 1) + 1):
        gamma = np.vdot(v_current, u_current)
        gamma_norm = abs(gamma)
        if gamma_norm < 1e-15:
            break

        signed_error = float(sigma_current) - float(epsilon)
        normal = gamma / gamma_norm
        z_trial = z_current - (signed_error / gamma_norm) * normal

        sigma_trial, u_trial, v_trial = svd_solver(A, z_trial)
        sigma_trial = float(sigma_trial)
        u_trial = np.asarray(u_trial, dtype=np.complex128)
        v_trial = np.asarray(v_trial, dtype=np.complex128)
        u_trial = u_trial / max(np.linalg.norm(u_trial), 1e-15)
        v_trial = v_trial / max(np.linalg.norm(v_trial), 1e-15)
        trial_error = abs(sigma_trial - float(epsilon))

        if trial_error >= current_error - 1e-14:
            break

        z_current = complex(z_trial)
        sigma_current = sigma_trial
        u_current = u_trial
        v_current = v_trial
        current_error = trial_error

        if current_error <= tol:
            return z_current, u_current, v_current, {
                'sigma': float(sigma_current),
                'sigma_error': float(current_error),
                'projection_distance': float(abs(z_current - z_candidate)),
                'projection_iterations': int(iteration),
                'projection_mode': 'local_normal_newton',
            }

    return None
