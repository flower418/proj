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
    ):
        self.A = np.asarray(A, dtype=np.complex128)
        self.epsilon = float(epsilon)
        self.ode_system = ode_system
        self.controller = controller
        self.svd_solver = svd_solver or smallest_singular_triplet
        self.fixed_step_size = fixed_step_size
        self.closure_tol = closure_tol

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

    def check_closure(self, z_current: complex, z_start: complex, current_step: int) -> bool:
        return current_step >= 10 and np.abs(z_current - z_start) < self.closure_tol

    def track(self, z0: complex, max_steps: int = 1000) -> Dict:
        u, v = self.initialize(z0)
        state = TrackerState(z=z0, u=u, v=v, prev_gamma_arg=None)
        trajectory = [z0]
        u_history = [u.copy()]
        v_history = [v.copy()]
        restart_indices = []
        step_sizes = []
        feature_history = []

        for step in range(max_steps):
            features = self.extract_state_features(state.z, state.u, state.v, prev_state=state)
            feature_history.append(features)
            if self.controller is not None:
                ds, need_restart = self.controller.predict(features)
            else:
                ds = self.fixed_step_size
                need_restart = False

            if need_restart:
                z, u, v = self.exact_svd_restart(state.z)
                restart_indices.append(step)
            else:
                z, u, v = rk4_triplet_step(
                    self.ode_system.get_full_derivatives,
                    state.z,
                    state.u,
                    state.v,
                    ds,
                )
                u = u / max(np.linalg.norm(u), 1e-15)
                v = v / max(np.linalg.norm(v), 1e-15)

            state = TrackerState(z=z, u=u, v=v, prev_gamma_arg=np.angle(np.vdot(u, v)))
            trajectory.append(z)
            u_history.append(u.copy())
            v_history.append(v.copy())
            step_sizes.append(float(ds))

            if self.check_closure(z, z0, current_step=step + 1):
                break

        return {
            "trajectory": np.asarray(trajectory, dtype=np.complex128),
            "u_history": u_history,
            "v_history": v_history,
            "restart_indices": restart_indices,
            "step_sizes": step_sizes,
            "feature_history": np.asarray(feature_history, dtype=np.float32),
        }
