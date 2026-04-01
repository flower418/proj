from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.nn.features import extract_features
from src.train.expert_solver import ExpertSolver


class DAggerAugmenter:
    """State-space perturbation with expert recovery labels."""

    def __init__(
        self,
        expert_solver: ExpertSolver,
        noise_scales: Optional[Dict[str, float]] = None,
    ):
        self.expert = expert_solver
        self.noise_scales = noise_scales or {
            "z_real": 1e-3,
            "z_imag": 1e-3,
            "u": 1e-2,
            "v": 1e-2,
            "phase": 0.1,
        }

    def perturb_state(self, z: complex, u: np.ndarray, v: np.ndarray, seed: Optional[int] = None) -> Tuple[complex, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        z_pert = z + rng.normal(0.0, self.noise_scales["z_real"]) + 1j * rng.normal(0.0, self.noise_scales["z_imag"])
        u_pert = u + rng.normal(0.0, self.noise_scales["u"], size=u.shape) + 1j * rng.normal(0.0, self.noise_scales["u"], size=u.shape)
        v_pert = v + rng.normal(0.0, self.noise_scales["v"], size=v.shape) + 1j * rng.normal(0.0, self.noise_scales["v"], size=v.shape)
        u_pert = u_pert / max(np.linalg.norm(u_pert), 1e-15)
        v_pert = v_pert / max(np.linalg.norm(v_pert), 1e-15)
        phase = rng.uniform(-self.noise_scales["phase"], self.noise_scales["phase"])
        phase_factor = np.exp(1j * phase)
        u_pert = u_pert * phase_factor
        v_pert = v_pert * phase_factor
        return z_pert, u_pert, v_pert

    def query_expert_at_perturbed_state(self, z_pert: complex, u_pert: np.ndarray, v_pert: np.ndarray, original_ds: float) -> Tuple[float, int, float, float]:
        result = self.expert._step_with_hint(
            z_pert,
            u_pert,
            v_pert,
            steps_since_restart=self.expert.min_steps_before_restart,
            first_step_hint=max(original_ds, 1e-8),
        )
        ds_value = max(result.ds_expert, 1e-8)
        return ds_value, result.y_restart, result.residual, result.sigma_error

    def augment_trajectory(self, trajectory: List[Dict], num_perturbations_per_point: int = 1) -> List[Dict]:
        augmented = []
        for i, point in enumerate(trajectory):
            if point["y_restart"] == 1:
                continue
            prev_gamma_arg = None if i == 0 else float(np.angle(np.vdot(trajectory[i - 1]["u"], trajectory[i - 1]["v"])))
            for j in range(num_perturbations_per_point):
                z_pert, u_pert, v_pert = self.perturb_state(point["z"], point["u"], point["v"], seed=i * 1000 + j)
                ds_recover, y_restart, residual, sigma_error = self.query_expert_at_perturbed_state(
                    z_pert, u_pert, v_pert, point["ds_expert"]
                )
                features = extract_features(
                    z=z_pert,
                    u=u_pert,
                    v=v_pert,
                    A=self.expert.A,
                    epsilon=self.expert.epsilon,
                    prev_gamma_arg=prev_gamma_arg,
                    prev_solver_iters=self.expert.ode.solver.get_iteration_count(),
                )
                augmented.append(
                    {
                        "z": z_pert,
                        "u": u_pert,
                        "v": v_pert,
                        "ds_expert": ds_recover,
                        "y_restart": y_restart,
                        "features": features,
                        "step": point.get("step", -1),
                        "residual": residual,
                        "sigma_error": sigma_error,
                        "source": "dagger",
                    }
                )
        return augmented
