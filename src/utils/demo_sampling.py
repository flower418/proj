from __future__ import annotations

import numpy as np


def generate_random_matrix(
    n: int,
    matrix_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if matrix_type == "complex":
        return rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    if matrix_type == "real":
        return rng.standard_normal((n, n))
    if matrix_type == "hermitian":
        base = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        return (base + base.conj().T) / 2.0
    raise ValueError(f"Unsupported matrix type: {matrix_type}")


def choose_point_around_random_eigenvalue(
    A: np.ndarray,
    rng: np.random.Generator,
    radius_range: tuple[float, float],
) -> tuple[complex, complex]:
    eigvals = np.linalg.eigvals(np.asarray(A, dtype=np.complex128))
    anchor = complex(eigvals[int(rng.integers(len(eigvals)))])
    spectral_scale = max(np.ptp(np.real(eigvals)), np.ptp(np.imag(eigvals)), 1.0)
    r_min, r_max = radius_range
    if not (0.0 < r_min < r_max):
        raise ValueError("radius_range must satisfy 0 < R_MIN < R_MAX.")
    radius = rng.uniform(r_min, r_max) * spectral_scale
    angle = rng.uniform(0.0, 2.0 * np.pi)
    z0 = anchor + radius * np.exp(1j * angle)
    return z0, anchor


def choose_point_in_spectral_box(
    A: np.ndarray,
    rng: np.random.Generator,
    padding: float,
) -> tuple[complex, complex]:
    eigvals = np.linalg.eigvals(np.asarray(A, dtype=np.complex128))
    x_min = float(np.min(np.real(eigvals)))
    x_max = float(np.max(np.real(eigvals)))
    y_min = float(np.min(np.imag(eigvals)))
    y_max = float(np.max(np.imag(eigvals)))
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    x_pad = max(float(padding), 0.0) * x_span
    y_pad = max(float(padding), 0.0) * y_span
    z_random = complex(
        rng.uniform(x_min - x_pad, x_max + x_pad),
        rng.uniform(y_min - y_pad, y_max + y_pad),
    )
    nearest = complex(eigvals[int(np.argmin(np.abs(eigvals - z_random)))])
    return z_random, nearest


def sample_random_point(
    A: np.ndarray,
    rng: np.random.Generator,
    point_sampler: str,
    radius_range: tuple[float, float],
    box_padding: float,
) -> tuple[complex, complex]:
    if point_sampler == "spectral_box":
        return choose_point_in_spectral_box(A, rng, padding=box_padding)
    if point_sampler == "around_eigenvalue":
        return choose_point_around_random_eigenvalue(A, rng, radius_range=radius_range)
    raise ValueError(f"Unsupported point_sampler: {point_sampler}")
