from __future__ import annotations

import numpy as np


def _drop_duplicate_endpoint(curve: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.complex128).reshape(-1)
    if len(curve) >= 2 and abs(curve[-1] - curve[0]) <= tol:
        return curve[:-1]
    return curve


def resample_curve_by_arclength(curve: np.ndarray, num_points: int = 512) -> np.ndarray:
    curve = _drop_duplicate_endpoint(curve)
    if num_points <= 0:
        raise ValueError("num_points must be positive.")
    if len(curve) == 0:
        return np.zeros((0,), dtype=np.complex128)
    if len(curve) == 1:
        return np.repeat(curve, num_points).astype(np.complex128)

    segment_lengths = np.abs(np.diff(curve))
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(cumulative[-1])
    if total_length <= 1e-15:
        return np.repeat(curve[:1], num_points).astype(np.complex128)

    targets = np.linspace(0.0, total_length, num_points, endpoint=False)
    real_interp = np.interp(targets, cumulative, np.real(curve))
    imag_interp = np.interp(targets, cumulative, np.imag(curve))
    return real_interp + 1j * imag_interp


def contour_distance_metrics(
    curve_a: np.ndarray,
    curve_b: np.ndarray,
    num_points: int = 512,
) -> dict[str, float]:
    a = resample_curve_by_arclength(curve_a, num_points=num_points)
    b = resample_curve_by_arclength(curve_b, num_points=num_points)
    if len(a) == 0 or len(b) == 0:
        return {
            "mean_nearest_distance": 0.0,
            "hausdorff_distance": 0.0,
        }

    pairwise = np.abs(a[:, None] - b[None, :])
    a_to_b = np.min(pairwise, axis=1)
    b_to_a = np.min(pairwise, axis=0)
    return {
        "mean_nearest_distance": float(0.5 * (np.mean(a_to_b) + np.mean(b_to_a))),
        "hausdorff_distance": float(max(np.max(a_to_b), np.max(b_to_a))),
    }
