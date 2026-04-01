from __future__ import annotations

from typing import Tuple

import numpy as np

from .pseudoinverse import PseudoinverseSolver


class ManifoldODE:
    """Implementation of the ODE system in Eq. (13)."""

    def __init__(self, A: np.ndarray, epsilon: float, solver: PseudoinverseSolver | None = None):
        self.A = np.asarray(A, dtype=np.complex128)
        self.epsilon = float(epsilon)
        self.n = self.A.shape[0]
        self.solver = solver or PseudoinverseSolver()
        self._cache_z: complex | None = None
        self._cache_M: np.ndarray | None = None
        self._cache_M_star: np.ndarray | None = None

    def compute_dz_ds(self, z: complex, u: np.ndarray, v: np.ndarray) -> complex:
        gamma = np.vdot(v, u)
        magnitude = np.abs(gamma)
        if magnitude < 1e-15:
            raise ValueError("v^*u is too small; contour tangent is ill-defined.")
        return 1j * gamma / magnitude

    def compute_dv_ds(self, z: complex, u: np.ndarray, v: np.ndarray, dz_ds: complex) -> np.ndarray:
        M = self._get_M(z)
        M_star = self._get_M_star(z)
        rhs = dz_ds * (M_star @ v) + self.epsilon * np.conj(dz_ds) * u
        H = lambda x: M_star @ (M @ x)
        dv_ds = -self.solver.solve(H, self.epsilon**2, rhs, null_vector=v)
        return self._enforce_gauge(dv_ds, v)

    def compute_du_ds(self, z: complex, u: np.ndarray, v: np.ndarray, dz_ds: complex) -> np.ndarray:
        M = self._get_M(z)
        M_star = self._get_M_star(z)
        rhs = np.conj(dz_ds) * (M @ u) + self.epsilon * dz_ds * v
        H = lambda x: M @ (M_star @ x)
        du_ds = -self.solver.solve(H, self.epsilon**2, rhs, null_vector=u)
        return self._enforce_gauge(du_ds, u)

    def get_full_derivatives(self, z: complex, u: np.ndarray, v: np.ndarray) -> Tuple[complex, np.ndarray, np.ndarray]:
        dz_ds = self.compute_dz_ds(z, u, v)
        du_ds = self.compute_du_ds(z, u, v, dz_ds)
        dv_ds = self.compute_dv_ds(z, u, v, dz_ds)
        return dz_ds, du_ds, dv_ds

    def _get_M(self, z: complex) -> np.ndarray:
        if self._cache_z is None or np.abs(z - self._cache_z) > 1e-14:
            self._cache_z = z
            self._cache_M = z * np.eye(self.n, dtype=np.complex128) - self.A
            self._cache_M_star = self._cache_M.conj().T
        return self._cache_M.copy()

    def _get_M_star(self, z: complex) -> np.ndarray:
        if self._cache_z is None or np.abs(z - self._cache_z) > 1e-14:
            self._cache_z = z
            self._cache_M = z * np.eye(self.n, dtype=np.complex128) - self.A
            self._cache_M_star = self._cache_M.conj().T
        return self._cache_M_star.copy()

    @staticmethod
    def _enforce_gauge(direction: np.ndarray, vector: np.ndarray) -> np.ndarray:
        phase_component = np.real(np.vdot(vector, direction))
        return direction - phase_component * vector
