from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch import nn


class NNController(nn.Module):
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dims: List[int] = [64, 64],
        dropout: float = 0.1,
        norm_type: str = "layernorm",
        step_size_min: float = 1e-6,
        step_size_max: float | None = None,
    ):
        super().__init__()
        self.step_size_min = step_size_min
        self.step_size_max = step_size_max
        self.norm_type = norm_type.lower()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            if self.norm_type == "batchnorm":
                norm_layer = nn.BatchNorm1d(h_dim)
            elif self.norm_type == "layernorm":
                norm_layer = nn.LayerNorm(h_dim)
            else:
                raise ValueError(f"Unsupported norm_type: {norm_type}")
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    norm_layer,
                ]
            )
            prev_dim = h_dim
        self.shared_encoder = nn.Sequential(*layers)
        self.step_head = nn.Sequential(nn.Linear(prev_dim, 1), nn.Softplus())
        self.restart_head = nn.Sequential(nn.Linear(prev_dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared_encoder(x)
        ds = self.step_head(features) + self.step_size_min
        if self.step_size_max is not None:
            ds = torch.clamp(ds, max=self.step_size_max)
        p_restart = self.restart_head(features)
        return ds, p_restart

    @torch.no_grad()
    def predict(self, state_np: np.ndarray) -> Tuple[float, bool]:
        self.eval()
        device = next(self.parameters()).device
        x = torch.as_tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
        ds, p_restart = self.forward(x)
        ds_value = float(ds.squeeze(0).item())
        need_restart = bool(p_restart.squeeze(0).item() > 0.5)
        return ds_value, need_restart
