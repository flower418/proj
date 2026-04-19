from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import brentq

from .svd import smallest_singular_triplet


def sigma_min_at(A: np.ndarray, z: complex) -> float:
    sigma, _, _ = smallest_singular_triplet(A, z)
    return float(sigma)


def _matrix_scale(A: np.ndarray) -> float:
    A = np.asarray(A, dtype=np.complex128)
    n = max(int(A.shape[0]), 1)
    return float(np.linalg.norm(A, ord="fro") / max(np.sqrt(float(n)), 1.0))


def _nearest_eigen_gap(eigvals: np.ndarray, anchor: complex) -> float:
    distances = np.abs(np.asarray(eigvals, dtype=np.complex128) - complex(anchor))
    valid = distances > 1e-10
    if not np.any(valid):
        return float("inf")
    return float(np.min(distances[valid]))


def _root_on_ray(
    A: np.ndarray,
    epsilon: float,
    center: complex,
    direction: complex,
    initial_radius: float,
    tol: float,
    max_radius: float = 1e4,
    max_expansions: int = 40,
) -> Tuple[complex, float]:
    direction = direction / max(abs(direction), 1e-15)

    def objective(radius: float) -> float:
        z = center + radius * direction
        return sigma_min_at(A, z) - epsilon

    left_r = 0.0
    left_v = objective(left_r)
    if abs(left_v) <= tol:
        return center, sigma_min_at(A, center)

    right_r = max(float(initial_radius), 1e-8)
    right_v = objective(right_r)
    expansions = 0
    while left_v * right_v > 0.0 and right_r < max_radius and expansions < max_expansions:
        right_r *= 2.0
        right_v = objective(right_r)
        expansions += 1

    if left_v * right_v > 0.0:
        raise ValueError("Unable to bracket the epsilon contour along the selected ray.")

    radius = brentq(objective, left_r, right_r, maxiter=200, xtol=tol)
    z = center + radius * direction
    return z, sigma_min_at(A, z)


def select_anchor_eigenvalue(A: np.ndarray, which: str = "rightmost") -> complex:
    eigvals = np.linalg.eigvals(np.asarray(A, dtype=np.complex128))
    if which == "rightmost":
        return eigvals[int(np.argmax(np.real(eigvals)))]
    if which == "leftmost":
        return eigvals[int(np.argmin(np.real(eigvals)))]
    if which == "topmost":
        return eigvals[int(np.argmax(np.imag(eigvals)))]
    if which == "bottommost":
        return eigvals[int(np.argmin(np.imag(eigvals)))]
    raise ValueError(f"Unsupported anchor mode: {which}")


def auto_select_contour_start(
    A: np.ndarray,
    epsilon: float,
    which: str = "rightmost",
    angle_offset: float = 0.0,
    tol: float = 1e-6,
) -> Tuple[complex, float, complex]:
    """Pick a deterministic contour start from an extreme eigenvalue."""
    center = complex(select_anchor_eigenvalue(A, which=which))
    base_direction = {
        "rightmost": 1.0 + 0.0j,
        "leftmost": -1.0 + 0.0j,
        "topmost": 0.0 + 1.0j,
        "bottommost": 0.0 - 1.0j,
    }[which]
    direction = base_direction * np.exp(1j * angle_offset)
    z0, sigma = _root_on_ray(
        np.asarray(A, dtype=np.complex128),
        epsilon=float(epsilon),
        center=center,
        direction=direction,
        initial_radius=max(10.0 * float(epsilon), 1e-3),
        tol=tol,
    )
    return z0, sigma, center


def auto_select_near_eigen_contour(
    A: np.ndarray,
    which: str = "rightmost",
    angle_offset: float = 0.0,
    gap_ratio: float = 0.18,
    fallback_radius_ratio: float = 0.05,
    min_radius_ratio: float = 1e-4,
    epsilon_floor_ratio: float = 1e-8,
) -> Tuple[complex, float, complex, float]:
    """Pick a start point on a contour intentionally close to an anchor eigenvalue.

    This is intended for inference/demo visualization: contours very far from the
    spectrum often look almost circular and are less informative.
    """

    A = np.asarray(A, dtype=np.complex128)
    eigvals = np.linalg.eigvals(A)
    center = complex(select_anchor_eigenvalue(A, which=which))
    matrix_scale = max(_matrix_scale(A), 1.0)
    nearest_gap = _nearest_eigen_gap(eigvals, center)

    if np.isfinite(nearest_gap):
        target_radius = float(max(gap_ratio, 1e-4) * nearest_gap)
    else:
        target_radius = float(max(fallback_radius_ratio, 1e-4) * matrix_scale)

    min_radius = float(max(min_radius_ratio, 1e-8) * matrix_scale)
    max_radius = float(max(0.25 * matrix_scale, min_radius))
    target_radius = float(np.clip(target_radius, min_radius, max_radius))

    base_direction = {
        "rightmost": 1.0 + 0.0j,
        "leftmost": -1.0 + 0.0j,
        "topmost": 0.0 + 1.0j,
        "bottommost": 0.0 - 1.0j,
    }[which]
    direction = base_direction * np.exp(1j * angle_offset)

    epsilon_floor = float(max(epsilon_floor_ratio, 1e-12) * matrix_scale)
    z0 = center + target_radius * direction
    epsilon = float(sigma_min_at(A, z0))

    expansions = 0
    while (not np.isfinite(epsilon) or epsilon <= epsilon_floor) and expansions < 8:
        target_radius = float(min(target_radius * 1.8, max_radius))
        z0 = center + target_radius * direction
        epsilon = float(sigma_min_at(A, z0))
        expansions += 1
        if target_radius >= max_radius - 1e-15:
            break

    return complex(z0), float(epsilon), center, float(target_radius)


def project_to_contour(
    A: np.ndarray,
    epsilon: float,
    z_guess: complex,
    tol: float = 1e-6,
) -> Tuple[complex, float]:
    """Project a user guess to the epsilon-pseudospectrum contour along a ray."""
    A = np.asarray(A, dtype=np.complex128)
    sigma_guess = sigma_min_at(A, z_guess)
    if abs(sigma_guess - epsilon) <= tol:
        return z_guess, sigma_guess

    eigvals = np.linalg.eigvals(A)
    center = eigvals[int(np.argmin(np.abs(eigvals - z_guess)))]
    direction = z_guess - center
    if abs(direction) < 1e-14:
        direction = 1.0 + 0.0j
    r0 = max(abs(z_guess - center), 1e-6)
    try:
        return _root_on_ray(
            A=A,
            epsilon=float(epsilon),
            center=center,
            direction=direction,
            initial_radius=r0,
            tol=tol,
        )
    except ValueError as exc:
        raise ValueError(
            "Unable to project the provided z0 onto the epsilon contour along the selected ray. "
            "Try a different z0 guess or enable automatic start-point selection."
        ) from exc
