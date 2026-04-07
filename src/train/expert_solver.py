from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq

from src.core.manifold_ode import ManifoldODE
from src.core.pseudoinverse import PseudoinverseSolver
from src.solvers.rk4 import rk4_triplet_step
from src.utils.contour_init import project_to_contour, sigma_min_at
from src.utils.local_projection import project_to_contour_by_local_normal
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
    backtracks: int = 0
    applied_projection: bool = False
    suggested_next_step: float = 0.0


class ExpertSolver:
    """High-accuracy teacher policy for precise contour-tracking data generation."""

    def __init__(
        self,
        A: np.ndarray,
        epsilon: float,
        rtol: float = 1e-8,
        atol: float = 1e-8,
        max_step: float = 0.1,
        first_step: float = 0.03,
        drift_threshold: float = 1e-4,
        closure_tol: float = 1e-3,
        min_steps_before_restart: int = 5,
        min_steps_before_closure: int = 32,
        min_winding_angle: float = 1.5 * np.pi,
        projection_tol: float = 1e-8,
        min_step_size: float = 1e-6,
        max_backtracks: int = 8,
        proposal_growth: float = 1.6,
        nominal_growth: float = 1.2,
        projection_growth: float = 1.05,
        mild_backtrack_shrink: float = 0.9,
        strong_backtrack_shrink: float = 0.7,
        solver: Optional[PseudoinverseSolver] = None,
        svd_solver=None,
    ):
        self.A = np.asarray(A, dtype=np.complex128)
        self.epsilon = float(epsilon)
        self.n = self.A.shape[0]
        self.rtol = rtol
        self.atol = atol
        self.max_step = float(max_step)
        self.first_step = float(first_step)
        self.drift_threshold = float(drift_threshold)
        self.closure_tol = float(closure_tol)
        self.min_steps_before_restart = int(min_steps_before_restart)
        self.min_steps_before_closure = int(min_steps_before_closure)
        self.min_winding_angle = float(min_winding_angle)
        self.projection_tol = float(projection_tol)
        self.min_step_size = float(min_step_size)
        self.max_backtracks = int(max_backtracks)
        self.proposal_growth = float(proposal_growth)
        self.nominal_growth = float(nominal_growth)
        self.projection_growth = float(projection_growth)
        self.mild_backtrack_shrink = float(mild_backtrack_shrink)
        self.strong_backtrack_shrink = float(strong_backtrack_shrink)
        self.svd_solver = svd_solver or smallest_singular_triplet
        self.linear_solver = solver or PseudoinverseSolver()
        self.ode = ManifoldODE(self.A, self.epsilon, solver=self.linear_solver)

    def compute_residual(self, z: complex, u: np.ndarray, v: np.ndarray) -> float:
        return residual_norm(self.A, z, u, v, self.epsilon)

    def compute_sigma_approx(self, z: complex, u: np.ndarray, v: np.ndarray) -> float:
        M = z * np.eye(self.n, dtype=np.complex128) - self.A
        return float(np.abs(np.vdot(u, M @ v)))

    def compute_sigma_error(self, z: complex, u: np.ndarray, v: np.ndarray) -> float:
        approx_error = abs(self.compute_sigma_approx(z, u, v) - self.epsilon)
        true_error = abs(sigma_min_at(self.A, z) - self.epsilon)
        return float(max(approx_error, true_error))

    def _closure_anchor(self, z0: complex) -> complex:
        eigvals = np.linalg.eigvals(self.A)
        return complex(eigvals[int(np.argmin(np.abs(eigvals - z0)))])

    def _check_closure(
        self,
        z_current: complex,
        z_start: complex,
        current_step: int,
        path_length: float,
        max_distance_from_start: float,
        winding_angle: float,
        last_step_size: float,
    ) -> bool:
        if current_step < self.min_steps_before_closure:
            return False

        effective_closure_tol = float(max(self.closure_tol, 0.5 * float(last_step_size)))
        if abs(z_current - z_start) >= effective_closure_tol:
            return False

        min_path_length = max(20.0 * self.closure_tol, 10.0 * self.first_step)
        min_escape_distance = max(10.0 * self.closure_tol, 5.0 * self.first_step)
        if path_length < min_path_length:
            return False
        if max_distance_from_start < min_escape_distance:
            return False
        if abs(winding_angle) < self.min_winding_angle:
            return False
        return True

    def _project_to_contour_locally(
        self,
        z_candidate: complex,
        search_radius: float,
    ) -> tuple[complex, np.ndarray, np.ndarray, dict] | None:
        del search_radius
        return project_to_contour_by_local_normal(
            A=self.A,
            epsilon=self.epsilon,
            z_candidate=z_candidate,
            svd_solver=self.svd_solver,
            projection_tol=self.projection_tol,
        )

    def _advance_projected_step(
        self,
        z: complex,
        u: np.ndarray,
        v: np.ndarray,
        ds: float,
    ) -> tuple[complex, np.ndarray, np.ndarray, float, dict]:
        ds_try = float(np.clip(ds, self.min_step_size, self.max_step))

        for backtrack_idx in range(self.max_backtracks + 1):
            z_candidate, _, _ = rk4_triplet_step(
                self.ode.get_full_derivatives,
                z,
                u,
                v,
                ds_try,
            )
            sigma_candidate, u_exact, v_exact = self.svd_solver(self.A, z_candidate)
            u_exact = u_exact / max(np.linalg.norm(u_exact), 1e-15)
            v_exact = v_exact / max(np.linalg.norm(v_exact), 1e-15)
            sigma_error = abs(float(sigma_candidate) - self.epsilon)

            if sigma_error <= max(self.projection_tol, 1e-10):
                return z_candidate, u_exact, v_exact, ds_try, {
                    "backtracks": backtrack_idx,
                    "applied_projection": False,
                    "sigma": float(sigma_candidate),
                    "sigma_error": float(sigma_error),
                    "projection_distance": 0.0,
                    "projection_expansions": 0,
                    "projection_mode": "none",
                }

            local_projection = self._project_to_contour_locally(z_candidate, search_radius=ds_try)
            if local_projection is not None:
                z_projected, u_projected, v_projected, projection_info = local_projection
                if projection_info["sigma_error"] <= max(self.projection_tol, 1e-8):
                    return z_projected, u_projected, v_projected, ds_try, {
                        "backtracks": backtrack_idx,
                        "applied_projection": True,
                        **projection_info,
                    }

            try:
                z_projected, sigma_projected = project_to_contour(
                    self.A,
                    self.epsilon,
                    z_candidate,
                    tol=min(self.projection_tol, 1e-6),
                )
            except ValueError:
                z_projected, sigma_projected = None, None
            if z_projected is not None and abs(float(sigma_projected) - self.epsilon) <= max(self.projection_tol, 1e-8):
                _, u_projected, v_projected = self.svd_solver(self.A, z_projected)
                u_projected = u_projected / max(np.linalg.norm(u_projected), 1e-15)
                v_projected = v_projected / max(np.linalg.norm(v_projected), 1e-15)
                return z_projected, u_projected, v_projected, ds_try, {
                    "backtracks": backtrack_idx,
                    "applied_projection": True,
                    "sigma": float(sigma_projected),
                    "sigma_error": float(abs(float(sigma_projected) - self.epsilon)),
                    "projection_distance": float(abs(z_projected - z_candidate)),
                    "projection_expansions": 0,
                    "projection_mode": "radial_fallback",
                }

            if ds_try <= self.min_step_size * (1.0 + 1e-12):
                break
            ds_try = max(0.5 * ds_try, self.min_step_size)

        raise RuntimeError("Unable to advance a teacher step while remaining on the epsilon contour.")

    def _propose_trial_step(self, step_hint: Optional[float]) -> float:
        if step_hint is None:
            return float(np.clip(self.first_step, self.min_step_size, self.max_step))
        base_step = max(float(step_hint), self.min_step_size)
        return float(np.clip(max(base_step * self.proposal_growth, self.first_step), self.min_step_size, self.max_step))

    def _adapt_next_step_size(self, current_step_size: float, backtracks: int, applied_projection: bool) -> float:
        next_step = float(current_step_size)
        if backtracks >= 2:
            next_step *= self.strong_backtrack_shrink
        elif backtracks == 1:
            next_step *= self.mild_backtrack_shrink
        elif applied_projection:
            next_step *= self.projection_growth
        else:
            next_step *= self.nominal_growth
        return float(np.clip(next_step, self.min_step_size, self.max_step))

    def _restart_reason(self, z: complex, u: np.ndarray, v: np.ndarray, steps_since_restart: int) -> Tuple[int, Optional[str], float, float, float]:
        residual = self.compute_residual(z, u, v)
        sigma_error = self.compute_sigma_error(z, u, v)
        gamma = abs(np.vdot(v, u))
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
                backtracks=0,
                applied_projection=False,
                suggested_next_step=max(self.first_step, self.min_step_size),
            )

        try:
            trial_step = self._propose_trial_step(first_step_hint)
            z_next, u_next, v_next, ds_expert, step_info = self._advance_projected_step(
                z=z,
                u=u,
                v=v,
                ds=trial_step,
            )
        except RuntimeError:
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
                restart_reason="projection_failed",
                backtracks=self.max_backtracks,
                applied_projection=False,
                suggested_next_step=max(self.first_step, self.min_step_size),
            )

        return ExpertStepResult(
            z_next=z_next,
            u_next=u_next,
            v_next=v_next,
            ds_expert=float(ds_expert),
            residual=residual,
            sigma_error=sigma_error,
            gamma=gamma,
            y_restart=0,
            restart_reason=None,
            backtracks=int(step_info["backtracks"]),
            applied_projection=bool(step_info["applied_projection"]),
            suggested_next_step=self._adapt_next_step_size(
                current_step_size=float(ds_expert),
                backtracks=int(step_info["backtracks"]),
                applied_projection=bool(step_info["applied_projection"]),
            ),
        )

    def generate_expert_trajectory(
        self,
        z0: complex,
        max_steps: int = 500,
        step_callback=None,
        max_wall_seconds: float | None = None,
    ) -> List[Dict]:
        _, u, v = self.svd_solver(self.A, z0)
        u = u / max(np.linalg.norm(u), 1e-15)
        v = v / max(np.linalg.norm(v), 1e-15)
        z = complex(z0)
        steps_since_restart = 0
        step_hint = self.first_step
        prev_ds = 0.0
        prev_applied_projection = False
        prev_applied_restart = False
        trajectory = []
        start_time = time.perf_counter()
        path_length = 0.0
        max_distance_from_start = 0.0
        closure_anchor = self._closure_anchor(z0)
        prev_anchor_angle = None if abs(z0 - closure_anchor) < 1e-12 else float(np.angle(z0 - closure_anchor))
        winding_angle = 0.0

        for step_idx in range(max_steps):
            z_prev = z
            result = self._step_with_hint(z, u, v, steps_since_restart=steps_since_restart, first_step_hint=step_hint)
            elapsed_seconds = float(time.perf_counter() - start_time)
            step_distance = float(abs(result.z_next - z_prev))
            path_length += step_distance
            max_distance_from_start = max(max_distance_from_start, float(abs(result.z_next - z0)))

            if prev_anchor_angle is not None and abs(result.z_next - closure_anchor) >= 1e-12:
                current_anchor_angle = float(np.angle(result.z_next - closure_anchor))
                delta = current_anchor_angle - prev_anchor_angle
                delta = float(np.angle(np.exp(1j * delta)))
                winding_angle += delta
                prev_anchor_angle = current_anchor_angle
            elif abs(result.z_next - closure_anchor) >= 1e-12:
                prev_anchor_angle = float(np.angle(result.z_next - closure_anchor))

            closed = False
            if result.y_restart == 0:
                closed = self._check_closure(
                    z_current=result.z_next,
                    z_start=z0,
                    current_step=step_idx + 1,
                    path_length=path_length,
                    max_distance_from_start=max_distance_from_start,
                    winding_angle=winding_angle,
                    last_step_size=max(float(result.ds_expert), self.min_step_size),
                )

            record = {
                "z": z_prev,
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
                "steps_since_restart": int(steps_since_restart),
                "prev_ds": float(prev_ds),
                "prev_applied_projection": bool(prev_applied_projection),
                "prev_applied_restart": bool(prev_applied_restart),
                "elapsed_seconds": elapsed_seconds,
                "backtracks": result.backtracks,
                "applied_projection": result.applied_projection,
                "step_distance": step_distance,
                "path_length": float(path_length),
                "max_distance_from_start": float(max_distance_from_start),
                "winding_angle": float(winding_angle),
                "closed": bool(closed),
            }
            trajectory.append(record)
            if step_callback is not None:
                step_callback(record)

            z, u, v = result.z_next, result.u_next, result.v_next
            prev_ds = 0.0 if result.y_restart else float(result.ds_expert)
            prev_applied_projection = bool(result.applied_projection)
            prev_applied_restart = bool(result.y_restart)
            steps_since_restart = 0 if result.y_restart else steps_since_restart + 1
            step_hint = (
                max(self.first_step, self.min_step_size)
                if result.y_restart
                else float(result.suggested_next_step)
            )

            if closed:
                break
            if max_wall_seconds is not None and elapsed_seconds >= float(max_wall_seconds):
                break
        return trajectory
