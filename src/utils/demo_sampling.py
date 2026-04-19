from __future__ import annotations

import numpy as np

from .contour_init import select_anchor_eigenvalue, sigma_min_at


SUPPORTED_MATRIX_TYPES = (
    "random_complex",
    "random_hermitian",
    "random_real",
    "ill_conditioned",
    "random_normal",
    "banded_nonnormal",
    "low_rank_plus_noise",
    "jordan_perturbed",
    "block_structured",
)

VISUAL_MATRIX_TYPES = (
    "banded_nonnormal",
    "jordan_perturbed",
    "low_rank_plus_noise",
    "block_structured",
    "random_complex",
)


class MatrixGenerator:
    @staticmethod
    def random_complex(n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))

    @staticmethod
    def random_hermitian(n: int, rng: np.random.Generator) -> np.ndarray:
        base = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        return (base + base.conj().T) / 2.0

    @staticmethod
    def random_real(n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal((n, n))

    @staticmethod
    def ill_conditioned(
        n: int,
        rng: np.random.Generator,
        condition_number: float = 1e6,
    ) -> np.ndarray:
        u, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
        v, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
        sigma = np.geomspace(1.0, 1.0 / condition_number, n)
        return u @ np.diag(sigma) @ v.conj().T

    @staticmethod
    def random_normal(n: int, rng: np.random.Generator) -> np.ndarray:
        q, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
        eigvals = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        return q @ np.diag(eigvals) @ q.conj().T

    @staticmethod
    def banded_nonnormal(n: int, rng: np.random.Generator) -> np.ndarray:
        diag = 0.2 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        upper = 0.9 * (rng.standard_normal(n - 1) + 1j * rng.standard_normal(n - 1)) if n > 1 else np.array([], dtype=np.complex128)
        lower = 0.15 * (rng.standard_normal(n - 1) + 1j * rng.standard_normal(n - 1)) if n > 1 else np.array([], dtype=np.complex128)
        matrix = np.diag(diag.astype(np.complex128))
        if n > 1:
            matrix += np.diag(upper.astype(np.complex128), 1)
            matrix += np.diag(lower.astype(np.complex128), -1)
        if n > 2:
            second_upper = 0.08 * (rng.standard_normal(n - 2) + 1j * rng.standard_normal(n - 2))
            matrix += np.diag(second_upper.astype(np.complex128), 2)
        return matrix

    @staticmethod
    def low_rank_plus_noise(n: int, rng: np.random.Generator) -> np.ndarray:
        rank = max(2, min(n // 6, 8))
        u = rng.standard_normal((n, rank)) + 1j * rng.standard_normal((n, rank))
        v = rng.standard_normal((n, rank)) + 1j * rng.standard_normal((n, rank))
        low_rank = (u @ v.conj().T) / max(np.sqrt(rank), 1.0)
        noise = 0.05 * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
        return low_rank + noise

    @staticmethod
    def jordan_perturbed(n: int, rng: np.random.Generator) -> np.ndarray:
        lam = complex(rng.standard_normal(), rng.standard_normal())
        matrix = lam * np.eye(n, dtype=np.complex128)
        if n > 1:
            matrix += np.diag(np.ones(n - 1, dtype=np.complex128), 1)
        matrix += 0.01 * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
        return matrix

    @staticmethod
    def block_structured(n: int, rng: np.random.Generator) -> np.ndarray:
        split = max(1, n // 2)
        matrix = np.zeros((n, n), dtype=np.complex128)
        matrix[:split, :split] = MatrixGenerator.random_complex(split, rng)
        if n - split > 0:
            block = MatrixGenerator.random_hermitian(n - split, rng)
            matrix[split:, split:] = block + (1.5 + 0.8j) * np.eye(n - split, dtype=np.complex128)
            matrix[:split, split:] = 0.08 * (rng.standard_normal((split, n - split)) + 1j * rng.standard_normal((split, n - split)))
            matrix[split:, :split] = 0.03 * (rng.standard_normal((n - split, split)) + 1j * rng.standard_normal((n - split, split)))
        return matrix


def sample_random_matrix_type(rng: np.random.Generator) -> str:
    return str(SUPPORTED_MATRIX_TYPES[int(rng.integers(len(SUPPORTED_MATRIX_TYPES)))])


def build_random_matrix(n: int, rng: np.random.Generator) -> tuple[str, np.ndarray]:
    matrix_type = sample_random_matrix_type(rng)
    matrix = getattr(MatrixGenerator, matrix_type)(n, rng)
    return matrix_type, np.asarray(matrix, dtype=np.complex128)


def _departure_from_normality(A: np.ndarray) -> float:
    A = np.asarray(A, dtype=np.complex128)
    AH = A.conj().T
    commutator = A @ AH - AH @ A
    scale = max(float(np.linalg.norm(A, ord="fro")) ** 2, 1e-12)
    return float(np.linalg.norm(commutator, ord="fro") / scale)


def build_visual_demo_matrix(
    n: int,
    rng: np.random.Generator,
    num_candidates: int = 6,
) -> tuple[str, np.ndarray]:
    best_type = None
    best_matrix = None
    best_score = -float("inf")
    candidate_count = max(int(num_candidates), 1)

    for _ in range(candidate_count):
        matrix_type = str(VISUAL_MATRIX_TYPES[int(rng.integers(len(VISUAL_MATRIX_TYPES)))])
        matrix = np.asarray(getattr(MatrixGenerator, matrix_type)(n, rng), dtype=np.complex128)
        score = _departure_from_normality(matrix)
        if score > best_score:
            best_score = score
            best_type = matrix_type
            best_matrix = matrix

    return str(best_type), np.asarray(best_matrix, dtype=np.complex128)


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


def sample_unrestricted_random_point(
    A: np.ndarray,
    rng: np.random.Generator,
    radius_range: tuple[float, float] = (0.25, 1.25),
) -> tuple[complex, complex]:
    eigvals = np.linalg.eigvals(np.asarray(A, dtype=np.complex128))
    center = complex(np.mean(eigvals))
    spectral_scale = max(np.ptp(np.real(eigvals)), np.ptp(np.imag(eigvals)), 1.0)
    low, high = radius_range
    radius = float(rng.uniform(low, high)) * spectral_scale
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    z_random = center + radius * np.exp(1j * angle)
    anchor = complex(eigvals[int(np.argmin(np.abs(eigvals - z_random)))])
    return complex(z_random), anchor


def sample_random_contour_start(
    A: np.ndarray,
    rng: np.random.Generator,
) -> tuple[complex, float, complex, complex]:
    z_random, anchor = sample_unrestricted_random_point(A=A, rng=rng)
    epsilon = float(sigma_min_at(A, z_random))
    return complex(z_random), epsilon, complex(z_random), complex(anchor)


def sample_near_eigen_contour_start(
    A: np.ndarray,
    rng: np.random.Generator,
    which: str | None = None,
    gap_ratio_range: tuple[float, float] = (0.06, 0.22),
    fallback_radius_ratio_range: tuple[float, float] = (0.02, 0.12),
    min_radius_ratio: float = 1e-4,
    max_radius_ratio: float = 0.25,
    epsilon_floor_ratio: float = 1e-8,
) -> tuple[complex, float, complex, complex, dict]:
    A = np.asarray(A, dtype=np.complex128)
    eigvals = np.linalg.eigvals(A)
    if which is None:
        anchor = complex(eigvals[int(rng.integers(len(eigvals)))])
        anchor_mode = "random_eigenvalue"
    else:
        anchor = complex(select_anchor_eigenvalue(A, which=which))
        anchor_mode = str(which)
    nearest_gap = _nearest_eigen_gap(eigvals, anchor)
    matrix_scale = max(_matrix_scale(A), 1.0)

    if np.isfinite(nearest_gap):
        low, high = gap_ratio_range
        radius = float(rng.uniform(low, high)) * nearest_gap
    else:
        low, high = fallback_radius_ratio_range
        radius = float(rng.uniform(low, high)) * matrix_scale

    min_radius = float(max(min_radius_ratio, 1e-8) * matrix_scale)
    max_radius = float(max(max_radius_ratio * matrix_scale, min_radius))
    radius = float(np.clip(radius, min_radius, max_radius))
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    direction = np.exp(1j * angle)
    z_random = complex(anchor + radius * direction)
    epsilon_floor = float(max(epsilon_floor_ratio, 1e-12) * matrix_scale)
    epsilon = float(sigma_min_at(A, z_random))

    expansions = 0
    while (not np.isfinite(epsilon) or epsilon <= epsilon_floor) and expansions < 8:
        radius = float(min(radius * 1.6, max_radius))
        z_random = complex(anchor + radius * direction)
        epsilon = float(sigma_min_at(A, z_random))
        expansions += 1
        if radius >= max_radius - 1e-15:
            break

    return z_random, epsilon, z_random, anchor, {
        "sampling_mode": "near_eigen",
        "sampling_radius": float(radius),
        "sampling_angle": float(angle),
        "anchor_mode": anchor_mode,
    }


def sample_training_contour_start(
    A: np.ndarray,
    rng: np.random.Generator,
    near_eigen_ratio: float = 0.75,
) -> tuple[complex, float, complex, complex, dict]:
    near_eigen_ratio = float(np.clip(near_eigen_ratio, 0.0, 1.0))
    if float(rng.random()) < near_eigen_ratio:
        return sample_near_eigen_contour_start(A=A, rng=rng)

    z0, epsilon, z_random, anchor = sample_random_contour_start(A=A, rng=rng)
    return z0, epsilon, z_random, anchor, {
        "sampling_mode": "random_point",
        "sampling_radius": float("nan"),
    }
