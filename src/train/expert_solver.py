from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.core.manifold_ode import ManifoldODE
from src.core.pseudoinverse import PseudoinverseSolver
from src.solvers.rk4 import rk4_triplet_step
from src.utils.contour_init import project_to_contour
from src.utils.local_projection import project_to_contour_by_local_normal
from src.utils.svd import smallest_singular_triplet


@dataclass
class ExpertStepSize:
    ds_expert: float
    suggested_next_step: float = 0.0


class ExpertSolver:
    """High-accuracy teacher policy for step-size supervision."""

    def __init__(
        self,
        A: np.ndarray,
        epsilon: float,
        max_step: float = 0.1,
        first_step: float = 0.03,
        projection_tol: float = 1e-8,
        min_step_size: float = 1e-6,
        proposal_growth: float = 1.6,
        nominal_growth: float = 1.2,
        projection_growth: float = 1.05,
        solver: Optional[PseudoinverseSolver] = None,
        svd_solver=None,
    ):
        self.A = np.asarray(A, dtype=np.complex128)
        self.epsilon = float(epsilon)
        self.max_step = float(max_step)
        self.first_step = float(first_step)
        self.projection_tol = float(projection_tol)
        self.min_step_size = float(min_step_size)
        self.proposal_growth = float(proposal_growth)
        self.nominal_growth = float(nominal_growth)
        self.projection_growth = float(projection_growth)
        self.svd_solver = svd_solver or smallest_singular_triplet
        self.linear_solver = solver or PseudoinverseSolver()
        self.ode = ManifoldODE(self.A, self.epsilon, solver=self.linear_solver)

    def _normalized_triplet(self, z: complex) -> tuple[np.ndarray, np.ndarray]:
        _, u, v = self.svd_solver(self.A, z)
        u = u / max(np.linalg.norm(u), 1e-15)
        v = v / max(np.linalg.norm(v), 1e-15)
        return u, v

    def _project_to_contour_locally(
        self,
        z_candidate: complex,
        sigma_candidate: float | None = None,
        u_candidate: np.ndarray | None = None,
        v_candidate: np.ndarray | None = None,
    ) -> tuple[complex, np.ndarray, np.ndarray, dict] | None:
        return project_to_contour_by_local_normal(
            A=self.A,
            epsilon=self.epsilon,
            z_candidate=z_candidate,
            svd_solver=self.svd_solver,
            projection_tol=self.projection_tol,
            sigma_current=sigma_candidate,
            u_current=u_candidate,
            v_current=v_candidate,
        )

    def _advance_projected_step(
        self,
        z: complex,
        u: np.ndarray,
        v: np.ndarray,
        ds: float,
    ) -> tuple[complex, np.ndarray, np.ndarray, float, dict]:
        ds_step = float(np.clip(ds, self.min_step_size, self.max_step))

        z_candidate, _, _ = rk4_triplet_step(
            self.ode.get_full_derivatives,
            z,
            u,
            v,
            ds_step,
        )
        sigma_candidate, u_exact, v_exact = self.svd_solver(self.A, z_candidate)
        sigma_candidate = float(sigma_candidate)
        u_exact = u_exact / max(np.linalg.norm(u_exact), 1e-15)
        v_exact = v_exact / max(np.linalg.norm(v_exact), 1e-15)
        sigma_error = abs(sigma_candidate - self.epsilon)

        if sigma_error <= max(self.projection_tol, 1e-10):
            return z_candidate, u_exact, v_exact, ds_step, {
                'applied_projection': False,
                'sigma': float(sigma_candidate),
                'sigma_error': float(sigma_error),
                'projection_distance': 0.0,
                'projection_iterations': 0,
                'projection_mode': 'none',
            }

        local_projection = self._project_to_contour_locally(
            z_candidate,
            sigma_candidate=float(sigma_candidate),
            u_candidate=u_exact,
            v_candidate=v_exact,
        )
        if local_projection is not None:
            z_projected, u_projected, v_projected, projection_info = local_projection
            if projection_info['sigma_error'] <= max(self.projection_tol, 1e-8):
                return z_projected, u_projected, v_projected, ds_step, {
                    'applied_projection': True,
                    **projection_info,
                }

        try:
            z_projected, sigma_projected = project_to_contour(
                self.A,
                self.epsilon,
                z_candidate,
                tol=min(self.projection_tol, 1e-6),
            )
        except ValueError as exc:
            raise RuntimeError('Unable to advance a teacher step while remaining on the epsilon contour.') from exc

        if abs(float(sigma_projected) - self.epsilon) > max(self.projection_tol, 1e-8):
            raise RuntimeError('Unable to advance a teacher step while remaining on the epsilon contour.')

        _, u_projected, v_projected = self.svd_solver(self.A, z_projected)
        u_projected = u_projected / max(np.linalg.norm(u_projected), 1e-15)
        v_projected = v_projected / max(np.linalg.norm(v_projected), 1e-15)
        return z_projected, u_projected, v_projected, ds_step, {
            'applied_projection': True,
            'sigma': float(sigma_projected),
            'sigma_error': float(abs(float(sigma_projected) - self.epsilon)),
            'projection_distance': float(abs(z_projected - z_candidate)),
            'projection_iterations': 0,
            'projection_mode': 'radial_fallback',
        }

    def _propose_trial_step(self, step_hint: Optional[float]) -> float:
        if step_hint is None:
            return float(np.clip(self.first_step, self.min_step_size, self.max_step))
        base_step = max(float(step_hint), self.min_step_size)
        return float(np.clip(max(base_step * self.proposal_growth, self.first_step), self.min_step_size, self.max_step))

    def _adapt_next_step_size(self, current_step_size: float, applied_projection: bool) -> float:
        next_step = float(current_step_size)
        if applied_projection:
            next_step *= self.projection_growth
        else:
            next_step *= self.nominal_growth
        return float(np.clip(next_step, self.min_step_size, self.max_step))

    def step_with_hint(
        self,
        z: complex,
        u: np.ndarray,
        v: np.ndarray,
        step_hint: Optional[float] = None,
    ) -> ExpertStepSize:
        gamma = float(abs(np.vdot(v, u)))

        step_u = u
        step_v = v
        if gamma < 1e-8:
            step_u, step_v = self._normalized_triplet(z)

        trial_step = self._propose_trial_step(step_hint)
        try:
            _, _, _, ds_expert, step_info = self._advance_projected_step(
                z=z,
                u=step_u,
                v=step_v,
                ds=trial_step,
            )
        except RuntimeError:
            u_exact, v_exact = self._normalized_triplet(z)
            retry_step = max(self.min_step_size, min(self.first_step, trial_step))
            _, _, _, ds_expert, step_info = self._advance_projected_step(
                z=z,
                u=u_exact,
                v=v_exact,
                ds=retry_step,
            )

        applied_projection = bool(step_info['applied_projection'])
        return ExpertStepSize(
            ds_expert=float(ds_expert),
            suggested_next_step=self._adapt_next_step_size(
                current_step_size=float(ds_expert),
                applied_projection=applied_projection,
            ),
        )
