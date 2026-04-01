from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from src.nn.features import extract_features
from src.train.dagger_augmentation import DAggerAugmenter
from src.train.expert_solver import ExpertSolver
from src.core.pseudoinverse import PseudoinverseSolver


class ExpertDataGenerator:
    def __init__(
        self,
        A: np.ndarray,
        epsilon: float,
        expert_tol: float = 1e-8,
        atol: float = 1e-8,
        drift_threshold: float = 1e-4,
        base_step_size: float = 1e-2,
        max_step_size: float = 0.1,
        closure_tol: float = 1e-3,
        solver: PseudoinverseSolver | None = None,
    ):
        self.A = np.asarray(A, dtype=np.complex128)
        self.epsilon = epsilon
        self.expert = ExpertSolver(
            A=self.A,
            epsilon=epsilon,
            rtol=expert_tol,
            atol=atol,
            max_step=max_step_size,
            first_step=base_step_size,
            drift_threshold=drift_threshold,
            closure_tol=closure_tol,
            solver=solver,
        )
        self.augmenter = DAggerAugmenter(self.expert)

    def generate_trajectory(self, z0: complex, max_steps: int = 500) -> List[Dict]:
        trajectory = self.expert.generate_expert_trajectory(z0=z0, max_steps=max_steps)
        records = []
        for idx, point in enumerate(trajectory):
            prev_gamma_arg = None if idx == 0 else float(np.angle(np.vdot(trajectory[idx - 1]["u"], trajectory[idx - 1]["v"])))
            features = extract_features(
                z=point["z"],
                u=point["u"],
                v=point["v"],
                A=self.A,
                epsilon=self.epsilon,
                prev_gamma_arg=prev_gamma_arg,
                prev_solver_iters=self.expert.ode.solver.get_iteration_count(),
            )
            record = dict(point)
            record["features"] = features
            record["source"] = "expert"
            records.append(record)
        return records

    def add_state_perturbations(
        self,
        trajectory: List[Dict],
        noise_std: float = 0.01,
        num_perturbations_per_point: int = 1,
    ) -> List[Dict]:
        scale = max(noise_std, 1e-12)
        self.augmenter.noise_scales = {
            "z_real": 1e-3 * scale / 0.01,
            "z_imag": 1e-3 * scale / 0.01,
            "u": scale,
            "v": scale,
            "phase": 0.1,
        }
        return self.augmenter.augment_trajectory(
            trajectory=trajectory,
            num_perturbations_per_point=num_perturbations_per_point,
        )


@dataclass
class ExpertRecord:
    features: np.ndarray
    ds_expert: float
    y_restart: int


class ExpertDataset(Dataset):
    def __init__(self, records: List[Dict]):
        self.records = [
            ExpertRecord(
                features=np.asarray(record["features"], dtype=np.float32),
                ds_expert=float(record["ds_expert"]),
                y_restart=int(record["y_restart"]),
            )
            for record in records
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        return {
            "features": torch.tensor(record.features, dtype=torch.float32),
            "ds_expert": torch.tensor(record.ds_expert, dtype=torch.float32),
            "y_restart": torch.tensor(record.y_restart, dtype=torch.float32),
        }
