from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def rk4_triplet_step(
    rhs: Callable[[complex, np.ndarray, np.ndarray], Tuple[complex, np.ndarray, np.ndarray]],
    z: complex,
    u: np.ndarray,
    v: np.ndarray,
    ds: float,
):
    k1z, k1u, k1v = rhs(z, u, v)
    k2z, k2u, k2v = rhs(z + 0.5 * ds * k1z, u + 0.5 * ds * k1u, v + 0.5 * ds * k1v)
    k3z, k3u, k3v = rhs(z + 0.5 * ds * k2z, u + 0.5 * ds * k2u, v + 0.5 * ds * k2v)
    k4z, k4u, k4v = rhs(z + ds * k3z, u + ds * k3u, v + ds * k3v)

    z_next = z + (ds / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)
    u_next = u + (ds / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u)
    v_next = v + (ds / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return z_next, u_next, v_next
