from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.nn.features import assemble_controller_features, extract_features
from src.train.expert_solver import ExpertSolver
from src.utils.contour_init import project_to_contour


PROGRESS_LOG_EVERY = 20


class DAggerAugmenter:
    """State-space perturbation with expert step-size recovery labels."""

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

    def query_expert_at_perturbed_state(
        self,
        z_pert: complex,
        u_pert: np.ndarray,
        v_pert: np.ndarray,
        original_ds: float,
    ) -> Tuple[float, float, float]:
        try:
            z_query, _ = project_to_contour(
                self.expert.A,
                self.expert.epsilon,
                z_pert,
                tol=min(self.expert.projection_tol, 1e-6),
            )
            u_query, v_query = self.expert._normalized_triplet(z_query)
        except ValueError:
            z_query = z_pert
            u_query = u_pert
            v_query = v_pert

        base_hint = max(float(original_ds), self.expert.first_step, self.expert.min_step_size)
        result = None
        last_error = None
        for factor in (1.0, 0.5, 0.25, 0.125):
            try:
                result = self.expert._step_with_hint(
                    z_query,
                    u_query,
                    v_query,
                    first_step_hint=max(base_hint * factor, self.expert.min_step_size),
                )
                break
            except RuntimeError as exc:
                last_error = exc
        if result is None:
            raise RuntimeError("Expert failed to recover a valid teacher step from perturbed state.") from last_error

        ds_value = max(result.ds_expert, 1e-8)
        return ds_value, result.residual, result.sigma_error

    def augment_trajectory(
        self,
        trajectory: List[Dict],
        num_perturbations_per_point: int = 1,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> List[Dict]:
        augmented = []
        total_queries = int(len(trajectory) * max(num_perturbations_per_point, 0))
        processed_queries = 0

        for i, point in enumerate(trajectory):
            for j in range(num_perturbations_per_point):
                z_pert, u_pert, v_pert = self.perturb_state(point["z"], point["u"], point["v"], seed=i * 1000 + j)
                ds_recover, residual, sigma_error = self.query_expert_at_perturbed_state(
                    z_pert,
                    u_pert,
                    v_pert,
                    point["ds_expert"],
                )
                base_features = extract_features(
                    z=z_pert,
                    u=u_pert,
                    v=v_pert,
                    A=self.expert.A,
                    epsilon=self.expert.epsilon,
                )
                input_dim = len(point["features"]) if "features" in point else None
                features = assemble_controller_features(
                    base_features,
                    prev_ds=float(point.get("prev_ds", 0.0)),
                    prev_applied_projection=bool(point.get("prev_applied_projection", False)),
                    input_dim=input_dim,
                )
                augmented.append(
                    {
                        "z": z_pert,
                        "u": u_pert,
                        "v": v_pert,
                        "ds_expert": ds_recover,
                        "features": features,
                        "step": point.get("step", -1),
                        "prev_ds": float(point.get("prev_ds", 0.0)),
                        "prev_applied_projection": bool(point.get("prev_applied_projection", False)),
                        "residual": residual,
                        "sigma_error": sigma_error,
                        "source": "dagger",
                    }
                )
                processed_queries += 1
                if progress_callback is not None and processed_queries % PROGRESS_LOG_EVERY == 0:
                    progress_callback(
                        {
                            "processed_queries": int(processed_queries),
                            "total_queries": int(total_queries),
                            "generated_samples": int(len(augmented)),
                            "trajectory_index": int(i + 1),
                            "perturbation_index": int(j + 1),
                        }
                    )
        return augmented
