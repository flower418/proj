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
    max_line_search_backtracks: int = 4,
):
    """Cheap local normal correction using damped Newton steps.

    This replaces the previous root-search along the local normal direction.
    It is much cheaper when most steps already land close to the contour.
    """

    tol = max(float(projection_tol), 1e-10)
    sigma_current, u_current, v_current = svd_solver(A, z_candidate)
    u_current = u_current / max(np.linalg.norm(u_current), 1e-15)
    v_current = v_current / max(np.linalg.norm(v_current), 1e-15)
    current_error = abs(float(sigma_current) - float(epsilon))
    if current_error <= tol:
        return z_candidate, u_current, v_current, {
            "sigma": float(sigma_current),
            "sigma_error": float(current_error),
            "projection_distance": 0.0,
            "projection_expansions": 0,
            "projection_mode": "local_exact",
        }

    z_current = complex(z_candidate)
    total_backtracks = 0

    for iteration in range(1, max(int(max_newton_iters), 1) + 1):
        signed_error = float(sigma_current) - float(epsilon)
        current_error = abs(signed_error)
        if current_error <= tol:
            return z_current, u_current, v_current, {
                "sigma": float(sigma_current),
                "sigma_error": float(current_error),
                "projection_distance": float(abs(z_current - z_candidate)),
                "projection_expansions": int(total_backtracks),
                "projection_mode": "local_normal_newton",
            }

        gamma = np.vdot(v_current, u_current)
        gamma_norm = abs(gamma)
        if gamma_norm < 1e-15:
            break
        normal = gamma / gamma_norm
        alpha = signed_error / gamma_norm

        accepted = False
        damping = 1.0
        for backtrack_idx in range(max(int(max_line_search_backtracks), 0) + 1):
            z_trial = z_current - damping * alpha * normal
            sigma_trial, u_trial, v_trial = svd_solver(A, z_trial)
            u_trial = u_trial / max(np.linalg.norm(u_trial), 1e-15)
            v_trial = v_trial / max(np.linalg.norm(v_trial), 1e-15)
            trial_error = abs(float(sigma_trial) - float(epsilon))
            if trial_error < current_error:
                z_current = complex(z_trial)
                sigma_current = float(sigma_trial)
                u_current = u_trial
                v_current = v_trial
                total_backtracks += backtrack_idx
                accepted = True
                break
            damping *= 0.5

        if not accepted:
            break

        if abs(float(sigma_current) - float(epsilon)) <= tol:
            return z_current, u_current, v_current, {
                "sigma": float(sigma_current),
                "sigma_error": float(abs(float(sigma_current) - float(epsilon))),
                "projection_distance": float(abs(z_current - z_candidate)),
                "projection_expansions": int(total_backtracks),
                "projection_mode": "local_normal_newton",
            }

    return None
