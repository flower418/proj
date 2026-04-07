from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def rk4_step(rhs, state, ds):
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * ds * k1)
    k3 = rhs(state + 0.5 * ds * k2)
    k4 = rhs(state + ds * k3)
    return state + (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


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


def heun_triplet_step(
    rhs: Callable[[complex, np.ndarray, np.ndarray], Tuple[complex, np.ndarray, np.ndarray]],
    z: complex,
    u: np.ndarray,
    v: np.ndarray,
    ds: float,
):
    k1z, k1u, k1v = rhs(z, u, v)
    z_pred = z + ds * k1z
    u_pred = u + ds * k1u
    v_pred = v + ds * k1v
    k2z, k2u, k2v = rhs(z_pred, u_pred, v_pred)

    z_next = z + 0.5 * ds * (k1z + k2z)
    u_next = u + 0.5 * ds * (k1u + k2u)
    v_next = v + 0.5 * ds * (k1v + k2v)
    return z_next, u_next, v_next
