from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from src.nn.features import extract_features
from src.solvers.rk4 import rk4_triplet_step
from src.utils.svd import smallest_singular_triplet


@dataclass
class TrackerState:
    z: complex
    u: np.ndarray
    v: np.ndarray
    prev_gamma_arg: float | None = None


class ContourTracker:
    def __init__(
        self,
        A: np.ndarray,
        epsilon: float,
        ode_system,
        controller: Optional[object] = None,
        svd_solver=None,
        fixed_step_size: float = 1e-2,
        closure_tol: float = 1e-3,
        min_steps_before_closure: int = 32,
        min_winding_angle: float = 1.5 * np.pi,
        min_steps_between_restarts: int = 5,
    ):
        self.A = np.asarray(A, dtype=np.complex128)
        self.epsilon = float(epsilon)
        self.ode_system = ode_system
        self.controller = controller
        self.svd_solver = svd_solver or smallest_singular_triplet
        self.fixed_step_size = fixed_step_size
        self.closure_tol = closure_tol
        self.min_steps_before_closure = int(min_steps_before_closure)
        self.min_winding_angle = float(min_winding_angle)
        self.min_steps_between_restarts = int(min_steps_between_restarts)

    def initialize(self, z0: complex) -> Tuple[np.ndarray, np.ndarray]:
        _, u0, v0 = self.svd_solver(self.A, z0)
        return u0, v0

    def exact_svd_restart(self, z: complex) -> Tuple[complex, np.ndarray, np.ndarray]:
        _, u, v = self.svd_solver(self.A, z)
        return z, u, v

    def extract_state_features(self, z, u, v, prev_state=None) -> np.ndarray:
        prev_gamma_arg = None if prev_state is None else prev_state.prev_gamma_arg
        prev_solver_iters = getattr(self.ode_system.solver, "get_iteration_count", lambda: 0)()
        return extract_features(
            z=z,
            u=u,
            v=v,
            A=self.A,
            epsilon=self.epsilon,
            prev_gamma_arg=prev_gamma_arg,
            prev_solver_iters=prev_solver_iters,
        )

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
    ) -> bool:
        if current_step < self.min_steps_before_closure:
            return False
        if np.abs(z_current - z_start) >= self.closure_tol:
            return False

        min_path_length = max(20.0 * self.closure_tol, 10.0 * self.fixed_step_size)
        min_escape_distance = max(10.0 * self.closure_tol, 5.0 * self.fixed_step_size)

        if path_length is not None and path_length < min_path_length:
            return False
        if max_distance_from_start is not None and max_distance_from_start < min_escape_distance:
            return False
        if winding_angle is not None and abs(winding_angle) < self.min_winding_angle:
            return False
        return True

    def track(self, z0: complex, max_steps: int = 1000) -> Dict:
        u, v = self.initialize(z0)
        state = TrackerState(z=z0, u=u, v=v, prev_gamma_arg=None)
        trajectory = [z0]
        u_history = [u.copy()]
        v_history = [v.copy()]
        restart_indices = []
        step_sizes = []
        feature_history = []
        path_length = 0.0
        max_distance_from_start = 0.0
        closure_anchor = self._closure_anchor(z0)
        prev_anchor_angle = None if abs(z0 - closure_anchor) < 1e-12 else float(np.angle(z0 - closure_anchor))
        winding_angle = 0.0
        closed = False
        steps_since_restart = self.min_steps_between_restarts

        for step in range(max_steps):
            features = self.extract_state_features(state.z, state.u, state.v, prev_state=state)
            feature_history.append(features)
            if self.controller is not None:
                ds, need_restart = self.controller.predict(features)
            else:
                ds = self.fixed_step_size
                need_restart = False
            ds = max(float(ds), 1e-12)

            if need_restart and steps_since_restart >= self.min_steps_between_restarts:
                _, u_step, v_step = self.exact_svd_restart(state.z)
                restart_indices.append(len(trajectory) - 1)
                steps_since_restart = 0
            else:
                u_step, v_step = state.u, state.v

            z, u, v = rk4_triplet_step(
                self.ode_system.get_full_derivatives,
                state.z,
                u_step,
                v_step,
                ds,
            )
            u = u / max(np.linalg.norm(u), 1e-15)
            v = v / max(np.linalg.norm(v), 1e-15)

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

            state = TrackerState(z=z, u=u, v=v, prev_gamma_arg=np.angle(np.vdot(u, v)))
            trajectory.append(z)
            u_history.append(u.copy())
            v_history.append(v.copy())
            step_sizes.append(float(ds))
            steps_since_restart += 1

            if self.check_closure(
                z,
                z0,
                current_step=step + 1,
                path_length=path_length,
                max_distance_from_start=max_distance_from_start,
                winding_angle=winding_angle,
            ):
                closed = True
                break

        return {
            "trajectory": np.asarray(trajectory, dtype=np.complex128),
            "u_history": u_history,
            "v_history": v_history,
            "restart_indices": restart_indices,
            "step_sizes": step_sizes,
            "feature_history": np.asarray(feature_history, dtype=np.float32),
            "closed": closed,
            "path_length": float(path_length),
            "max_distance_from_start": float(max_distance_from_start),
            "winding_angle": float(winding_angle),
            "closure_anchor": closure_anchor,
        }
