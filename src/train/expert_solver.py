from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import RK45

from src.core.manifold_ode import ManifoldODE
from src.core.pseudoinverse import PseudoinverseSolver
from src.utils.metrics import residual_norm
from src.utils.svd import smallest_singular_triplet


@dataclass
class ExpertStepResult:
    z_next: complex
    u_next: np.ndarray
    v_next: np.ndarray
    ds_expert: float
    residual: float
    sigma_error: float
    gamma: float
    y_restart: int
    restart_reason: Optional[str] = None


class ExpertSolver:
    """High-accuracy teacher policy based on single-step adaptive RK45."""

    def __init__(
        self,
        A: np.ndarray,
        epsilon: float,
        rtol: float = 1e-8,
        atol: float = 1e-8,
        max_step: float = 0.1,
        first_step: float = 0.01,
        drift_threshold: float = 1e-4,
        closure_tol: float = 1e-3,
        min_steps_before_restart: int = 5,
        solver: Optional[PseudoinverseSolver] = None,
        svd_solver=None,
    ):
        self.A = np.asarray(A, dtype=np.complex128)
        self.epsilon = float(epsilon)
        self.n = self.A.shape[0]
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step
        self.first_step = first_step
        self.drift_threshold = drift_threshold
        self.closure_tol = closure_tol
        self.min_steps_before_restart = min_steps_before_restart
        self.svd_solver = svd_solver or smallest_singular_triplet
        self.linear_solver = solver or PseudoinverseSolver()
        self.ode = ManifoldODE(self.A, self.epsilon, solver=self.linear_solver)

    def compute_residual(self, z: complex, u: np.ndarray, v: np.ndarray) -> float:
        return residual_norm(self.A, z, u, v, self.epsilon)

    def compute_sigma_approx(self, z: complex, u: np.ndarray, v: np.ndarray) -> float:
        M = z * np.eye(self.n, dtype=np.complex128) - self.A
        return float(np.abs(np.vdot(u, M @ v)))

    def _pack_state(self, z: complex, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        y = np.zeros(2 + 4 * self.n, dtype=np.float64)
        y[0] = np.real(z)
        y[1] = np.imag(z)
        y[2 : 2 + self.n] = np.real(u)
        y[2 + self.n : 2 + 2 * self.n] = np.imag(u)
        y[2 + 2 * self.n : 2 + 3 * self.n] = np.real(v)
        y[2 + 3 * self.n :] = np.imag(v)
        return y

    def _unpack_state(self, y: np.ndarray) -> Tuple[complex, np.ndarray, np.ndarray]:
        z = y[0] + 1j * y[1]
        u = y[2 : 2 + self.n] + 1j * y[2 + self.n : 2 + 2 * self.n]
        v = y[2 + 2 * self.n : 2 + 3 * self.n] + 1j * y[2 + 3 * self.n :]
        return z, u, v

    def _ode_rhs(self, _: float, y: np.ndarray) -> np.ndarray:
        z, u, v = self._unpack_state(y)
        dz_ds, du_ds, dv_ds = self.ode.get_full_derivatives(z, u, v)
        dy = np.zeros_like(y, dtype=np.float64)
        dy[0] = np.real(dz_ds)
        dy[1] = np.imag(dz_ds)
        dy[2 : 2 + self.n] = np.real(du_ds)
        dy[2 + self.n : 2 + 2 * self.n] = np.imag(du_ds)
        dy[2 + 2 * self.n : 2 + 3 * self.n] = np.real(dv_ds)
        dy[2 + 3 * self.n :] = np.imag(dv_ds)
        return dy

    def _restart_reason(self, z: complex, u: np.ndarray, v: np.ndarray, steps_since_restart: int) -> Tuple[int, Optional[str], float, float, float]:
        residual = self.compute_residual(z, u, v)
        sigma_error = abs(self.compute_sigma_approx(z, u, v) - self.epsilon)
        gamma = abs(np.vdot(u, v))
        if gamma < 1e-6:
            return 1, "singular_gamma", residual, sigma_error, gamma
        if steps_since_restart >= self.min_steps_before_restart and residual > self.drift_threshold:
            return 1, "drift", residual, sigma_error, gamma
        if steps_since_restart >= self.min_steps_before_restart and sigma_error > self.drift_threshold:
            return 1, "sigma_error", residual, sigma_error, gamma
        return 0, None, residual, sigma_error, gamma

    def step(self, z: complex, u: np.ndarray, v: np.ndarray, steps_since_restart: int = 0) -> ExpertStepResult:
        return self._step_with_hint(z, u, v, steps_since_restart=steps_since_restart, first_step_hint=None)

    def _step_with_hint(
        self,
        z: complex,
        u: np.ndarray,
        v: np.ndarray,
        steps_since_restart: int = 0,
        first_step_hint: Optional[float] = None,
    ) -> ExpertStepResult:
        y_restart, restart_reason, residual, sigma_error, gamma = self._restart_reason(z, u, v, steps_since_restart)
        if y_restart:
            _, u_exact, v_exact = self.svd_solver(self.A, z)
            u_exact = u_exact / max(np.linalg.norm(u_exact), 1e-15)
            v_exact = v_exact / max(np.linalg.norm(v_exact), 1e-15)
            return ExpertStepResult(
                z_next=z,
                u_next=u_exact,
                v_next=v_exact,
                ds_expert=0.0,
                residual=residual,
                sigma_error=sigma_error,
                gamma=gamma,
                y_restart=1,
                restart_reason=restart_reason,
            )

        y0 = self._pack_state(z, u, v)
        rk_solver = RK45(
            fun=self._ode_rhs,
            t0=0.0,
            y0=y0,
            t_bound=1.0,
            max_step=self.max_step,
            rtol=self.rtol,
            atol=self.atol,
            first_step=min(first_step_hint or self.first_step, self.max_step),
        )
        rk_solver.step()
        if rk_solver.status == "failed":
            _, u_exact, v_exact = self.svd_solver(self.A, z)
            u_exact = u_exact / max(np.linalg.norm(u_exact), 1e-15)
            v_exact = v_exact / max(np.linalg.norm(v_exact), 1e-15)
            return ExpertStepResult(
                z_next=z,
                u_next=u_exact,
                v_next=v_exact,
                ds_expert=0.0,
                residual=residual,
                sigma_error=sigma_error,
                gamma=gamma,
                y_restart=1,
                restart_reason="rk_failure",
            )
        ds_expert = float(rk_solver.step_size if rk_solver.step_size is not None else self.first_step)
        z_next, u_next, v_next = self._unpack_state(rk_solver.y)
        u_next = u_next / max(np.linalg.norm(u_next), 1e-15)
        v_next = v_next / max(np.linalg.norm(v_next), 1e-15)
        return ExpertStepResult(
            z_next=z_next,
            u_next=u_next,
            v_next=v_next,
            ds_expert=ds_expert,
            residual=residual,
            sigma_error=sigma_error,
            gamma=gamma,
            y_restart=0,
        )

    def generate_expert_trajectory(self, z0: complex, max_steps: int = 500) -> List[Dict]:
        _, u, v = self.svd_solver(self.A, z0)
        u = u / max(np.linalg.norm(u), 1e-15)
        v = v / max(np.linalg.norm(v), 1e-15)
        z = z0
        steps_since_restart = 0
        step_hint = self.first_step
        trajectory = []

        for step_idx in range(max_steps):
            result = self._step_with_hint(z, u, v, steps_since_restart=steps_since_restart, first_step_hint=step_hint)
            trajectory.append(
                {
                    "z": z,
                    "u": u.copy(),
                    "v": v.copy(),
                    "z_next": result.z_next,
                    "u_next": result.u_next.copy(),
                    "v_next": result.v_next.copy(),
                    "ds_expert": result.ds_expert,
                    "y_restart": result.y_restart,
                    "restart_reason": result.restart_reason,
                    "residual": result.residual,
                    "sigma_error": result.sigma_error,
                    "gamma": result.gamma,
                    "step": step_idx,
                }
            )
            z, u, v = result.z_next, result.u_next, result.v_next
            steps_since_restart = 0 if result.y_restart else steps_since_restart + 1
            step_hint = self.first_step if result.y_restart else max(min(result.ds_expert, self.max_step), 1e-8)
            if step_idx >= 10 and abs(z - z0) < self.closure_tol:
                break
        return trajectory
