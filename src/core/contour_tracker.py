from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from src.nn.features import assemble_controller_features, extract_features
from src.utils.contour_init import project_to_contour, sigma_min_at
from src.utils.local_projection import project_to_contour_by_local_normal
from src.utils.svd import smallest_singular_triplet


@dataclass
class TrackerState:
    z: complex
    u: np.ndarray
    v: np.ndarray
    prev_ds: float = 0.0


class ContourTracker:
    TANGENT_OVERLAP_TOL = 1.0e-10

    def __init__(
        self,
        A: np.ndarray,
        epsilon: float,
        controller: Optional[object] = None,
        svd_solver=None,
        fixed_step_size: float = 1e-2,
        closure_tol: float = 1e-3,
        min_steps_before_closure: int = 32,
        min_winding_angle: float = 1.5 * np.pi,
        projection_tol: float = 1e-4,
        min_step_size: float = 1e-6,
        projection_defer_factor: float = 4.0,
        projection_defer_distance_ratio: float = 0.08,
        max_deferred_projection_steps: int = 6,
        exact_triplet_refresh_interval: int = 8,
        approx_triplet_sigma_tol: float | None = 5.0e-3,
        approx_triplet_residual_tol: float | None = 5.0e-2,
    ):
        self.A = np.asarray(A, dtype=np.complex128)
        self.epsilon = float(epsilon)
        self.controller = controller
        self.svd_solver = svd_solver or smallest_singular_triplet
        self.fixed_step_size = float(fixed_step_size)
        self.closure_tol = float(closure_tol)
        self.min_steps_before_closure = int(min_steps_before_closure)
        self.min_winding_angle = float(min_winding_angle)
        self.projection_tol = float(projection_tol)
        self.min_step_size = float(min_step_size)
        self.projection_defer_factor = max(float(projection_defer_factor), 1.0)
        self.projection_defer_distance_ratio = max(float(projection_defer_distance_ratio), 0.0)
        self.max_deferred_projection_steps = max(int(max_deferred_projection_steps), 0)
        self.exact_triplet_refresh_interval = max(int(exact_triplet_refresh_interval), 0)
        self.approx_triplet_sigma_tol = None if approx_triplet_sigma_tol is None else max(float(approx_triplet_sigma_tol), 0.0)
        self.approx_triplet_residual_tol = None if approx_triplet_residual_tol is None else max(float(approx_triplet_residual_tol), 0.0)

    def initialize(self, z0: complex) -> Tuple[np.ndarray, np.ndarray]:
        _, u0, v0 = self.svd_solver(self.A, z0)
        return u0, v0

    def refresh_triplet(self, z: complex) -> Tuple[complex, np.ndarray, np.ndarray]:
        _, u, v = self.svd_solver(self.A, z)
        return z, u, v

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        return np.asarray(vector, dtype=np.complex128) / max(np.linalg.norm(vector), 1e-15)

    def _refresh_normalized_triplet(self, z: complex) -> tuple[np.ndarray, np.ndarray]:
        _, u, v = self.refresh_triplet(z)
        return self._normalize_vector(u), self._normalize_vector(v)

    def _ensure_well_defined_tangent_state(
        self,
        z: complex,
        u: np.ndarray,
        v: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if abs(np.vdot(v, u)) >= self.TANGENT_OVERLAP_TOL:
            return self._normalize_vector(u), self._normalize_vector(v)
        u_exact, v_exact = self._refresh_normalized_triplet(z)
        if abs(np.vdot(v_exact, u_exact)) < self.TANGENT_OVERLAP_TOL:
            raise ValueError('v^*u is too small; contour tangent is ill-defined.')
        return u_exact, v_exact

    def _compute_tangent_direction(
        self,
        u: np.ndarray,
        v: np.ndarray,
        z: complex | None = None,
        preferred_direction: complex | None = None,
    ) -> complex:
        gamma = np.vdot(v, u)
        magnitude = np.abs(gamma)
        if magnitude >= 1e-12:
            tangent = 1j * gamma / magnitude
            if preferred_direction is not None and abs(preferred_direction) > 1e-12:
                preferred = complex(preferred_direction) / abs(preferred_direction)
                if float(np.real(np.conj(preferred) * tangent)) < 0.0:
                    tangent = -tangent
            return tangent
        raise ValueError('v^*u is too small; contour tangent is ill-defined.')

    def extract_state_features(self, z, u, v, prev_state=None) -> np.ndarray:
        base_features = extract_features(
            z=z,
            u=u,
            v=v,
            A=self.A,
            epsilon=self.epsilon,
        )
        controller_model = getattr(self.controller, 'base_controller', self.controller)
        input_dim = int(getattr(controller_model, 'input_dim', len(base_features) + 1))
        prev_ds = 0.0 if prev_state is None else float(prev_state.prev_ds)
        return assemble_controller_features(
            base_features,
            prev_ds=prev_ds,
            input_dim=input_dim,
        )

    def _closure_anchor(self, z0: complex) -> complex:
        eigvals = np.linalg.eigvals(self.A)
        return complex(eigvals[int(np.argmin(np.abs(eigvals - z0)))])

    def _effective_closure_tol(self, last_step_size: float | None = None) -> float:
        return float(
            max(
                self.closure_tol,
                0.5 * float(last_step_size) if last_step_size is not None else 0.0,
            )
        )

    @staticmethod
    def _segment_distance_to_point(z_prev: complex, z_current: complex, point: complex) -> float:
        segment = complex(z_current - z_prev)
        segment_norm_sq = float(np.abs(segment) ** 2)
        if segment_norm_sq <= 1e-24:
            return float(np.abs(point - z_current))
        t = float(np.real(np.conj(segment) * (point - z_prev)) / segment_norm_sq)
        t = float(np.clip(t, 0.0, 1.0))
        closest = z_prev + t * segment
        return float(np.abs(closest - point))

    def check_closure(
        self,
        z_current: complex,
        z_start: complex,
        current_step: int,
        path_length: float | None = None,
        max_distance_from_start: float | None = None,
        winding_angle: float | None = None,
        last_step_size: float | None = None,
        z_prev: complex | None = None,
    ) -> bool:
        if current_step < self.min_steps_before_closure:
            return False
        min_path_length = max(20.0 * self.closure_tol, 10.0 * self.fixed_step_size)
        min_escape_distance = max(10.0 * self.closure_tol, 5.0 * self.fixed_step_size)
        if path_length is not None and path_length < min_path_length:
            return False
        if max_distance_from_start is not None and max_distance_from_start < min_escape_distance:
            return False
        if winding_angle is not None and abs(winding_angle) < self.min_winding_angle:
            return False

        effective_closure_tol = self._effective_closure_tol(last_step_size)
        if np.abs(z_current - z_start) < effective_closure_tol:
            return True
        if z_prev is None:
            return False
        return self._segment_distance_to_point(z_prev, z_current, z_start) < effective_closure_tol

    def _project_initial_point(self, z0: complex) -> tuple[complex, float]:
        sigma0 = sigma_min_at(self.A, z0)
        if abs(sigma0 - self.epsilon) <= max(self.projection_tol, self.closure_tol):
            return z0, float(sigma0)
        return project_to_contour(self.A, self.epsilon, z0, tol=min(self.projection_tol, 1e-6))

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

    def _predict_candidate_state(
        self,
        z: complex,
        u: np.ndarray,
        v: np.ndarray,
        ds: float,
        preferred_tangent: complex | None = None,
    ) -> tuple[complex, np.ndarray, np.ndarray]:
        u_step, v_step = self._ensure_well_defined_tangent_state(z, u, v)
        dz_ds = self._compute_tangent_direction(
            u_step,
            v_step,
            z=z,
            preferred_direction=preferred_tangent,
        )
        return z + ds * dz_ds, u_step.copy(), v_step.copy()

    def _approximate_triplet_metrics(
        self,
        z_candidate: complex,
        u_candidate: np.ndarray,
        v_candidate: np.ndarray,
    ) -> tuple[float, float]:
        Mv = z_candidate * v_candidate - self.A @ v_candidate
        sigma_approx = float(abs(np.vdot(u_candidate, Mv)))
        residual = float(np.linalg.norm(Mv - self.epsilon * u_candidate))
        return sigma_approx, residual

    @staticmethod
    def _estimate_projection_distance_from_triplet(
        sigma_error: float,
        u_candidate: np.ndarray,
        v_candidate: np.ndarray,
    ) -> float:
        gamma_norm = abs(np.vdot(v_candidate, u_candidate))
        return float(sigma_error / max(gamma_norm, 1e-12))

    def _advance_step(
        self,
        z: complex,
        u: np.ndarray,
        v: np.ndarray,
        ds: float,
        deferred_projection_streak: int = 0,
        steps_since_exact_triplet_refresh: int = 0,
        preferred_tangent: complex | None = None,
    ) -> Tuple[complex, np.ndarray, np.ndarray, float, dict]:
        ds_step = max(float(ds), self.min_step_size)
        z_candidate, u_candidate, v_candidate = self._predict_candidate_state(
            z=z,
            u=u,
            v=v,
            ds=ds_step,
            preferred_tangent=preferred_tangent,
        )

        can_use_approx_triplet = (
            self.exact_triplet_refresh_interval > 0
            and steps_since_exact_triplet_refresh < self.exact_triplet_refresh_interval
            and self.approx_triplet_sigma_tol is not None
            and self.approx_triplet_residual_tol is not None
        )
        if can_use_approx_triplet:
            sigma_approx, residual_approx = self._approximate_triplet_metrics(
                z_candidate=z_candidate,
                u_candidate=u_candidate,
                v_candidate=v_candidate,
            )
            sigma_error_approx = abs(sigma_approx - self.epsilon)
            residual_limit = max(float(self.approx_triplet_residual_tol), 1.25 * max(ds_step, self.min_step_size))
            approx_projection_distance = self._estimate_projection_distance_from_triplet(
                sigma_error=sigma_error_approx,
                u_candidate=u_candidate,
                v_candidate=v_candidate,
            )
            relaxed_projection_tol = self.projection_tol * self.projection_defer_factor

            if sigma_error_approx <= self.approx_triplet_sigma_tol and residual_approx <= residual_limit:
                return z_candidate, u_candidate, v_candidate, ds_step, {
                    'applied_projection': False,
                    'sigma': float(sigma_approx),
                    'sigma_error': float(sigma_error_approx),
                    'raw_sigma': float(sigma_approx),
                    'raw_sigma_error': float(sigma_error_approx),
                    'projection_distance': 0.0,
                    'projection_iterations': 0,
                    'projection_mode': 'none',
                    'triplet_refresh_mode': 'approx_skip',
                    'triplet_residual': float(residual_approx),
                    'exact_triplet_refresh': False,
                }

            allow_approx_deferred_projection = (
                self.max_deferred_projection_steps > 0
                and deferred_projection_streak < self.max_deferred_projection_steps
                and sigma_error_approx <= relaxed_projection_tol
                and approx_projection_distance <= self.projection_defer_distance_ratio * max(ds_step, self.min_step_size)
                and residual_approx <= 2.0 * residual_limit
            )
            if allow_approx_deferred_projection:
                return z_candidate, u_candidate, v_candidate, ds_step, {
                    'applied_projection': False,
                    'sigma': float(sigma_approx),
                    'sigma_error': float(sigma_error_approx),
                    'raw_sigma': float(sigma_approx),
                    'raw_sigma_error': float(sigma_error_approx),
                    'projection_distance': 0.0,
                    'projection_iterations': 0,
                    'projection_mode': 'deferred_accept',
                    'estimated_projection_distance': float(approx_projection_distance),
                    'triplet_refresh_mode': 'approx_deferred_accept',
                    'triplet_residual': float(residual_approx),
                    'exact_triplet_refresh': False,
                }

            approx_projection_window = max(6.0 * float(self.approx_triplet_sigma_tol), 3.0 * relaxed_projection_tol)
            if sigma_error_approx <= approx_projection_window and residual_approx <= 3.0 * residual_limit:
                local_projection = self._project_to_contour_locally(
                    z_candidate,
                    sigma_candidate=float(sigma_approx),
                    u_candidate=u_candidate,
                    v_candidate=v_candidate,
                )
                if local_projection is not None:
                    z_projected, u_projected, v_projected, projection_info = local_projection
                    if projection_info['sigma_error'] <= max(self.projection_tol, 1e-8):
                        return z_projected, u_projected, v_projected, ds_step, {
                            'applied_projection': True,
                            'raw_sigma': float(sigma_approx),
                            'raw_sigma_error': float(sigma_error_approx),
                            **projection_info,
                            'triplet_refresh_mode': 'approx_local_projection',
                            'triplet_residual': float(residual_approx),
                            'exact_triplet_refresh': True,
                        }

        sigma_candidate, u_exact, v_exact = self.svd_solver(self.A, z_candidate)
        sigma_candidate = float(sigma_candidate)
        u_exact = u_exact / max(np.linalg.norm(u_exact), 1e-15)
        v_exact = v_exact / max(np.linalg.norm(v_exact), 1e-15)
        sigma_error = abs(sigma_candidate - self.epsilon)

        if sigma_error <= self.projection_tol:
            return z_candidate, u_exact, v_exact, ds_step, {
                'applied_projection': False,
                'sigma': float(sigma_candidate),
                'sigma_error': float(sigma_error),
                'raw_sigma': float(sigma_candidate),
                'raw_sigma_error': float(sigma_error),
                'projection_distance': 0.0,
                'projection_iterations': 0,
                'projection_mode': 'none',
                'triplet_refresh_mode': 'exact_svd',
                'triplet_residual': 0.0,
                'exact_triplet_refresh': True,
            }

        estimated_projection_distance = self._estimate_projection_distance_from_triplet(
            sigma_error=sigma_error,
            u_candidate=u_exact,
            v_candidate=v_exact,
        )
        relaxed_projection_tol = self.projection_tol * self.projection_defer_factor
        allow_deferred_projection = (
            self.max_deferred_projection_steps > 0
            and deferred_projection_streak < self.max_deferred_projection_steps
            and sigma_error <= relaxed_projection_tol
            and estimated_projection_distance <= self.projection_defer_distance_ratio * max(ds_step, self.min_step_size)
        )
        if allow_deferred_projection:
            return z_candidate, u_exact, v_exact, ds_step, {
                'applied_projection': False,
                'sigma': float(sigma_candidate),
                'sigma_error': float(sigma_error),
                'raw_sigma': float(sigma_candidate),
                'raw_sigma_error': float(sigma_error),
                'projection_distance': 0.0,
                'projection_iterations': 0,
                'projection_mode': 'deferred_accept',
                'estimated_projection_distance': float(estimated_projection_distance),
                'triplet_refresh_mode': 'exact_svd',
                'triplet_residual': 0.0,
                'exact_triplet_refresh': True,
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
                    'raw_sigma': float(sigma_candidate),
                    'raw_sigma_error': float(sigma_error),
                    **projection_info,
                    'triplet_refresh_mode': 'local_projection',
                    'triplet_residual': 0.0,
                    'exact_triplet_refresh': True,
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
            _, u_projected, v_projected = self.refresh_triplet(z_projected)
            u_projected = u_projected / max(np.linalg.norm(u_projected), 1e-15)
            v_projected = v_projected / max(np.linalg.norm(v_projected), 1e-15)
            return z_projected, u_projected, v_projected, ds_step, {
                'applied_projection': True,
                'sigma': float(sigma_projected),
                'sigma_error': float(abs(float(sigma_projected) - self.epsilon)),
                'raw_sigma': float(sigma_candidate),
                'raw_sigma_error': float(sigma_error),
                'projection_distance': float(abs(z_projected - z_candidate)),
                'projection_iterations': 0,
                'projection_mode': 'radial_fallback',
                'triplet_refresh_mode': 'radial_projection',
                'triplet_residual': 0.0,
                'exact_triplet_refresh': True,
            }

        return z_candidate, u_exact, v_exact, ds_step, {
            'applied_projection': False,
            'sigma': float(sigma_candidate),
            'sigma_error': float(sigma_error),
            'raw_sigma': float(sigma_candidate),
            'raw_sigma_error': float(sigma_error),
            'projection_distance': 0.0,
            'projection_iterations': 0,
            'projection_mode': 'exact_fallback',
            'triplet_refresh_mode': 'exact_svd_fallback',
            'triplet_residual': 0.0,
            'exact_triplet_refresh': True,
        }

    def track(self, z0: complex, max_steps: int = 1000, step_callback=None) -> Dict:
        if self.controller is not None and hasattr(self.controller, 'reset'):
            self.controller.reset()

        z0, sigma_at_start = self._project_initial_point(z0)
        u, v = self.initialize(z0)
        state = TrackerState(
            z=z0,
            u=u,
            v=v,
            prev_ds=0.0,
        )
        trajectory = [z0]
        u_history = [u.copy()]
        v_history = [v.copy()]
        step_sizes = []
        path_length = 0.0
        max_distance_from_start = 0.0
        closure_anchor = self._closure_anchor(z0)
        prev_anchor_angle = None if abs(z0 - closure_anchor) < 1e-12 else float(np.angle(z0 - closure_anchor))
        winding_angle = 0.0
        closed = False
        projection_indices = []
        deferred_projection_streak = 0
        steps_since_exact_triplet_refresh = 0
        failure_reason = None
        try:
            u, v = self._ensure_well_defined_tangent_state(z0, u, v)
            state = TrackerState(
                z=z0,
                u=u,
                v=v,
                prev_ds=0.0,
            )
            u_history[0] = u.copy()
            v_history[0] = v.copy()
            prev_tangent = self._compute_tangent_direction(u, v, z=z0)
        except Exception:
            prev_tangent = None

        for step in range(max_steps):
            z_prev = state.z
            try:
                state.u, state.v = self._ensure_well_defined_tangent_state(state.z, state.u, state.v)
                u_history[-1] = state.u.copy()
                v_history[-1] = state.v.copy()
            except ValueError:
                failure_reason = 'ill_defined_tangent'
                break
            features = self.extract_state_features(state.z, state.u, state.v, prev_state=state)
            if self.controller is not None:
                if hasattr(self.controller, 'predict_with_info'):
                    ds, controller_info = self.controller.predict_with_info(features)
                else:
                    ds = self.controller.predict(features)
                    controller_info = {}
            else:
                ds = self.fixed_step_size
                controller_info = {}

            raw_ds = max(float(ds), 1e-12)
            ds = max(raw_ds, self.min_step_size)

            try:
                z, u, v, accepted_ds, step_diagnostics = self._advance_step(
                    state.z,
                    state.u,
                    state.v,
                    ds=ds,
                    deferred_projection_streak=deferred_projection_streak,
                    steps_since_exact_triplet_refresh=steps_since_exact_triplet_refresh,
                    preferred_tangent=prev_tangent,
                )
            except ValueError:
                failure_reason = 'ill_defined_tangent'
                break
            sigma_after_correction = float(step_diagnostics['sigma'])
            sigma_error_after_correction = float(step_diagnostics['sigma_error'])
            raw_sigma = float(step_diagnostics.get('raw_sigma', sigma_after_correction))
            raw_sigma_error = float(step_diagnostics.get('raw_sigma_error', sigma_error_after_correction))
            applied_projection = bool(step_diagnostics['applied_projection'])
            if applied_projection:
                projection_indices.append(len(trajectory))
                deferred_projection_streak = 0
            elif step_diagnostics.get('projection_mode') == 'deferred_accept':
                deferred_projection_streak += 1
            else:
                deferred_projection_streak = 0
            if bool(step_diagnostics.get('exact_triplet_refresh', True)):
                steps_since_exact_triplet_refresh = 0
            else:
                steps_since_exact_triplet_refresh += 1

            step_distance = float(np.abs(z - state.z))
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

            tangent_turn = 0.0
            try:
                current_tangent = self._compute_tangent_direction(
                    u,
                    v,
                    z=z,
                    preferred_direction=prev_tangent,
                )
            except Exception:
                current_tangent = prev_tangent
            if prev_tangent is not None and current_tangent is not None:
                tangent_turn = float(abs(np.angle(np.exp(1j * (np.angle(current_tangent) - np.angle(prev_tangent))))))
            prev_tangent = current_tangent

            state = TrackerState(
                z=z,
                u=u,
                v=v,
                prev_ds=float(accepted_ds),
            )
            trajectory.append(z)
            u_history.append(u.copy())
            v_history.append(v.copy())
            step_sizes.append(float(accepted_ds))

            if step_callback is not None:
                step_callback(
                    {
                        'step': step,
                        'z_prev': z_prev,
                        'z_next': z,
                        'raw_ds': float(raw_ds),
                        'ds': float(accepted_ds),
                        'step_distance': step_distance,
                        'distance_to_start': float(np.abs(z - z0)),
                        'path_length': float(path_length),
                        'max_distance_from_start': float(max_distance_from_start),
                        'winding_angle': float(winding_angle),
                        'sigma': float(sigma_after_correction),
                        'sigma_error': float(sigma_error_after_correction),
                        'raw_sigma': float(raw_sigma),
                        'raw_sigma_error': float(raw_sigma_error),
                        'applied_projection': applied_projection,
                        'projection_distance': float(step_diagnostics.get('projection_distance', 0.0)),
                        'projection_iterations': int(step_diagnostics.get('projection_iterations', 0)),
                        'projection_mode': step_diagnostics.get('projection_mode', 'none'),
                        'estimated_projection_distance': float(step_diagnostics.get('estimated_projection_distance', 0.0)),
                        'triplet_refresh_mode': step_diagnostics.get('triplet_refresh_mode', 'exact_svd'),
                        'triplet_residual': float(step_diagnostics.get('triplet_residual', 0.0)),
                        'steps_since_exact_triplet_refresh': int(steps_since_exact_triplet_refresh),
                        'tangent_turn': float(tangent_turn),
                        'controller_info': controller_info,
                        'features': features.copy(),
                    }
                )
            if self.controller is not None and hasattr(self.controller, 'observe_step'):
                self.controller.observe_step(
                    {
                        'step': step,
                        'ds': float(accepted_ds),
                        'applied_projection': applied_projection,
                        'sigma_error': float(sigma_error_after_correction),
                        'raw_sigma_error': float(raw_sigma_error),
                        'projection_distance': float(step_diagnostics.get('projection_distance', 0.0)),
                        'tangent_turn': float(tangent_turn),
                        'controller_info': controller_info,
                    }
                )

            if self.check_closure(
                z,
                z0,
                current_step=step + 1,
                path_length=path_length,
                max_distance_from_start=max_distance_from_start,
                winding_angle=winding_angle,
                last_step_size=accepted_ds,
                z_prev=z_prev,
            ):
                if np.abs(z - z0) >= self._effective_closure_tol(accepted_ds):
                    _, u_closure, v_closure = self.refresh_triplet(z0)
                    trajectory[-1] = z0
                    u_history[-1] = u_closure.copy()
                    v_history[-1] = v_closure.copy()
                closed = True
                break

        return {
            'trajectory': np.asarray(trajectory, dtype=np.complex128),
            'u_history': u_history,
            'v_history': v_history,
            'step_sizes': step_sizes,
            'closed': closed,
            'path_length': float(path_length),
            'max_distance_from_start': float(max_distance_from_start),
            'winding_angle': float(winding_angle),
            'closure_anchor': closure_anchor,
            'projection_indices': projection_indices,
            'sigma_at_start': float(sigma_at_start),
            'failure_reason': failure_reason,
        }
