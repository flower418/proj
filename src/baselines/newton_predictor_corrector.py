from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from src.utils.contour_init import project_to_contour, sigma_min_at
from src.utils.svd import smallest_singular_triplet


@dataclass
class NewtonCorrectorResult:
    z: complex
    u: np.ndarray
    v: np.ndarray
    sigma: float
    iterations: int
    line_search_shrinks: int
    converged: bool


class NewtonPredictorCorrectorTracker:
    """Classic tangent predictor plus Newton normal corrector baseline."""

    def __init__(
        self,
        A: np.ndarray,
        epsilon: float,
        initial_step_size: float = 1e-2,
        min_step_size: float = 1e-6,
        max_step_size: float = 1e-1,
        corrector_tol: float = 1e-10,
        max_corrector_iters: int = 8,
        max_step_halvings: int = 8,
        max_line_search_shrinks: int = 8,
        closure_tol: float = 1e-3,
        min_steps_before_closure: int = 32,
        min_winding_angle: float = 1.5 * np.pi,
        svd_solver=None,
    ):
        if not (0.0 < min_step_size <= initial_step_size <= max_step_size):
            raise ValueError("Step sizes must satisfy 0 < min_step_size <= initial_step_size <= max_step_size.")
        self.A = np.asarray(A, dtype=np.complex128)
        self.epsilon = float(epsilon)
        self.initial_step_size = float(initial_step_size)
        self.min_step_size = float(min_step_size)
        self.max_step_size = float(max_step_size)
        self.corrector_tol = float(corrector_tol)
        self.max_corrector_iters = int(max_corrector_iters)
        self.max_step_halvings = int(max_step_halvings)
        self.max_line_search_shrinks = int(max_line_search_shrinks)
        self.closure_tol = float(closure_tol)
        self.min_steps_before_closure = int(min_steps_before_closure)
        self.min_winding_angle = float(min_winding_angle)
        self.svd_solver = svd_solver or smallest_singular_triplet

    def _closure_anchor(self, z0: complex) -> complex:
        eigvals = np.linalg.eigvals(self.A)
        return complex(eigvals[int(np.argmin(np.abs(eigvals - z0)))])

    def check_closure(
        self,
        z_current: complex,
        z_start: complex,
        current_step: int,
        path_length: float | None = None,
        max_distance_from_start: float | None = None,
        winding_angle: float | None = None,
        last_step_size: float | None = None,
    ) -> bool:
        if current_step < self.min_steps_before_closure:
            return False
        effective_closure_tol = float(
            max(
                self.closure_tol,
                0.5 * float(last_step_size) if last_step_size is not None else 0.0,
            )
        )
        if np.abs(z_current - z_start) >= effective_closure_tol:
            return False

        min_path_length = max(20.0 * self.closure_tol, 10.0 * self.initial_step_size)
        min_escape_distance = max(10.0 * self.closure_tol, 5.0 * self.initial_step_size)

        if path_length is not None and path_length < min_path_length:
            return False
        if max_distance_from_start is not None and max_distance_from_start < min_escape_distance:
            return False
        if winding_angle is not None and abs(winding_angle) < self.min_winding_angle:
            return False
        return True

    def _initialize_state(self, z0: complex) -> Tuple[complex, np.ndarray, np.ndarray, float]:
        sigma0 = sigma_min_at(self.A, z0)
        if abs(sigma0 - self.epsilon) > max(self.corrector_tol, self.closure_tol):
            z0, sigma0 = project_to_contour(self.A, self.epsilon, z0, tol=min(self.corrector_tol, 1e-6))
        sigma0, u0, v0 = self.svd_solver(self.A, z0)
        u0 = u0 / max(np.linalg.norm(u0), 1e-15)
        v0 = v0 / max(np.linalg.norm(v0), 1e-15)
        return z0, u0, v0, float(sigma0)

    @staticmethod
    def _tangent_direction(u: np.ndarray, v: np.ndarray) -> complex:
        gamma = np.vdot(v, u)
        gamma_norm = abs(gamma)
        if gamma_norm < 1e-15:
            raise ValueError("v^*u is too small; tangent direction is ill-defined.")
        return 1j * gamma / gamma_norm

    def _newton_correct(
        self,
        z_pred: complex,
    ) -> NewtonCorrectorResult:
        z_current = complex(z_pred)
        total_line_search_shrinks = 0

        sigma, u, v = self.svd_solver(self.A, z_current)
        u = u / max(np.linalg.norm(u), 1e-15)
        v = v / max(np.linalg.norm(v), 1e-15)

        for iteration in range(1, self.max_corrector_iters + 1):
            sigma_error = float(sigma - self.epsilon)
            if abs(sigma_error) <= self.corrector_tol:
                return NewtonCorrectorResult(
                    z=z_current,
                    u=u,
                    v=v,
                    sigma=float(sigma),
                    iterations=iteration,
                    line_search_shrinks=total_line_search_shrinks,
                    converged=True,
                )

            gamma = np.vdot(v, u)
            gamma_norm = abs(gamma)
            if gamma_norm < 1e-15:
                break
            normal = gamma / gamma_norm
            alpha = sigma_error / gamma_norm

            accepted = False
            damping = 1.0
            for shrink_idx in range(self.max_line_search_shrinks + 1):
                z_trial = z_current - damping * alpha * normal
                sigma_trial, u_trial, v_trial = self.svd_solver(self.A, z_trial)
                if abs(sigma_trial - self.epsilon) < abs(sigma_error):
                    z_current = z_trial
                    sigma = float(sigma_trial)
                    u = u_trial / max(np.linalg.norm(u_trial), 1e-15)
                    v = v_trial / max(np.linalg.norm(v_trial), 1e-15)
                    total_line_search_shrinks += shrink_idx
                    accepted = True
                    break
                damping *= 0.5

            if not accepted:
                break

        return NewtonCorrectorResult(
            z=z_current,
            u=u,
            v=v,
            sigma=float(sigma),
            iterations=self.max_corrector_iters,
            line_search_shrinks=total_line_search_shrinks,
            converged=abs(float(sigma) - self.epsilon) <= self.corrector_tol,
        )

    def _adapt_step_size(
        self,
        current_step_size: float,
        corrector_iterations: int,
        halving_count: int,
    ) -> float:
        next_step = float(current_step_size)
        if halving_count > 0:
            next_step *= 0.9
        elif corrector_iterations <= 2:
            next_step *= 1.25
        elif corrector_iterations >= max(4, self.max_corrector_iters - 2):
            next_step *= 0.7
        return float(np.clip(next_step, self.min_step_size, self.max_step_size))

    def track(self, z0: complex, max_steps: int = 4000, step_callback=None) -> Dict:
        z0, u, v, sigma_at_start = self._initialize_state(z0)
        z = z0
        trajectory = [z0]
        u_history = [u.copy()]
        v_history = [v.copy()]
        step_sizes = []
        sigma_errors = [float(abs(sigma_at_start - self.epsilon))]
        corrector_iterations = []
        predictor_halvings = []
        line_search_shrinks = []
        path_length = 0.0
        max_distance_from_start = 0.0
        closure_anchor = self._closure_anchor(z0)
        prev_anchor_angle = None if abs(z0 - closure_anchor) < 1e-12 else float(np.angle(z0 - closure_anchor))
        winding_angle = 0.0
        closed = False
        step_size = self.initial_step_size
        failure_reason = None

        for step in range(max_steps):
            tangent = self._tangent_direction(u, v)
            accepted = None
            accepted_step = None
            accepted_halvings = None

            for halving_count in range(self.max_step_halvings + 1):
                trial_step = max(step_size * (0.5 ** halving_count), self.min_step_size)
                z_pred = z + trial_step * tangent
                corrector = self._newton_correct(z_pred)
                if corrector.converged:
                    accepted = corrector
                    accepted_step = float(trial_step)
                    accepted_halvings = int(halving_count)
                    break
                if trial_step <= self.min_step_size * (1.0 + 1e-12):
                    break

            if accepted is None or accepted_step is None or accepted_halvings is None:
                failure_reason = "corrector_failed"
                break

            z_prev = z
            z = accepted.z
            u = accepted.u
            v = accepted.v

            step_distance = float(np.abs(z - z_prev))
            path_length += step_distance
            max_distance_from_start = max(max_distance_from_start, float(np.abs(z - z0)))

            if prev_anchor_angle is not None and abs(z - closure_anchor) >= 1e-12:
                current_anchor_angle = float(np.angle(z - closure_anchor))
                delta = current_anchor_angle - prev_anchor_angle
                delta = float(np.angle(np.exp(1j * delta)))
                winding_angle += delta
                prev_anchor_angle = current_anchor_angle
            elif abs(z - closure_anchor) >= 1e-12:
                prev_anchor_angle = float(np.angle(z - closure_anchor))

            trajectory.append(z)
            u_history.append(u.copy())
            v_history.append(v.copy())
            step_sizes.append(float(accepted_step))
            sigma_errors.append(float(abs(accepted.sigma - self.epsilon)))
            corrector_iterations.append(int(accepted.iterations))
            predictor_halvings.append(int(accepted_halvings))
            line_search_shrinks.append(int(accepted.line_search_shrinks))

            if step_callback is not None:
                step_callback(
                    {
                        "step": step,
                        "z_prev": z_prev,
                        "z_next": z,
                        "ds": float(accepted_step),
                        "step_distance": step_distance,
                        "distance_to_start": float(np.abs(z - z0)),
                        "path_length": float(path_length),
                        "max_distance_from_start": float(max_distance_from_start),
                        "winding_angle": float(winding_angle),
                        "sigma": float(accepted.sigma),
                        "sigma_error": float(abs(accepted.sigma - self.epsilon)),
                        "predictor_halvings": int(accepted_halvings),
                        "corrector_iterations": int(accepted.iterations),
                        "line_search_shrinks": int(accepted.line_search_shrinks),
                    }
                )

            if self.check_closure(
                z_current=z,
                z_start=z0,
                current_step=step + 1,
                path_length=path_length,
                max_distance_from_start=max_distance_from_start,
                winding_angle=winding_angle,
                last_step_size=accepted_step,
            ):
                closed = True
                break

            step_size = self._adapt_step_size(
                current_step_size=accepted_step,
                corrector_iterations=accepted.iterations,
                halving_count=accepted_halvings,
            )

        return {
            "trajectory": np.asarray(trajectory, dtype=np.complex128),
            "u_history": u_history,
            "v_history": v_history,
            "step_sizes": step_sizes,
            "sigma_errors": sigma_errors,
            "corrector_iterations": corrector_iterations,
            "predictor_halvings": predictor_halvings,
            "line_search_shrinks": line_search_shrinks,
            "projection_indices": [],
            "closed": bool(closed),
            "path_length": float(path_length),
            "max_distance_from_start": float(max_distance_from_start),
            "winding_angle": float(winding_angle),
            "closure_anchor": closure_anchor,
            "sigma_at_start": float(sigma_at_start),
            "failure_reason": failure_reason,
            "final_step_size": float(step_size),
            "mean_corrector_iterations": float(np.mean(corrector_iterations)) if corrector_iterations else 0.0,
            "mean_line_search_shrinks": float(np.mean(line_search_shrinks)) if line_search_shrinks else 0.0,
            "mean_predictor_halvings": float(np.mean(predictor_halvings)) if predictor_halvings else 0.0,
        }
