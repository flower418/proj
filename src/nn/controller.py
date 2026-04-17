from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn


class NNController(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        norm_type: str = "layernorm",
        activation: str = "silu",
        head_hidden_dim: int | None = None,
        step_size_min: float = 1e-6,
        step_size_max: float | None = None,
    ):
        super().__init__()
        hidden_dims = list(hidden_dims or [64, 64])
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size.")
        if step_size_max is not None and step_size_max <= step_size_min:
            raise ValueError("step_size_max must be greater than step_size_min.")

        self.input_dim = int(input_dim)
        self.hidden_dims = hidden_dims
        self.dropout = float(dropout)
        self.step_size_min = float(step_size_min)
        self.step_size_max = None if step_size_max is None else float(step_size_max)
        self.norm_type = norm_type.lower()
        self.activation = activation.lower()
        self.head_hidden_dim = head_hidden_dim

        layers = []
        prev_dim = self.input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    self._make_norm(h_dim),
                    self._make_activation(),
                    nn.Dropout(self.dropout),
                ]
            )
            prev_dim = h_dim

        self.shared_encoder = nn.Sequential(*layers)
        self.step_head = self._make_head(prev_dim)

        if self.step_size_max is not None:
            self._log_step_min = math.log(self.step_size_min)
            self._log_step_span = math.log(self.step_size_max) - self._log_step_min

    def _make_norm(self, dim: int) -> nn.Module:
        if self.norm_type == "batchnorm":
            return nn.BatchNorm1d(dim)
        if self.norm_type == "layernorm":
            return nn.LayerNorm(dim)
        raise ValueError(f"Unsupported norm_type: {self.norm_type}")

    def _make_activation(self) -> nn.Module:
        if self.activation == "relu":
            return nn.ReLU()
        if self.activation == "gelu":
            return nn.GELU()
        if self.activation == "silu":
            return nn.SiLU()
        raise ValueError(f"Unsupported activation: {self.activation}")

    def _make_head(self, input_dim: int) -> nn.Sequential:
        layers = []
        hidden_dim = self.head_hidden_dim
        if hidden_dim is not None and hidden_dim > 0:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    self._make_activation(),
                    nn.Dropout(self.dropout),
                    nn.Linear(hidden_dim, 1),
                ]
            )
        else:
            layers.append(nn.Linear(input_dim, 1))
        return nn.Sequential(*layers)

    def get_config(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "norm_type": self.norm_type,
            "activation": self.activation,
            "head_hidden_dim": self.head_hidden_dim,
            "step_size_min": self.step_size_min,
            "step_size_max": self.step_size_max,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.shared_encoder(x)
        raw_ds = self.step_head(features)
        if self.step_size_max is not None:
            return torch.exp(self._log_step_min + torch.sigmoid(raw_ds) * self._log_step_span)
        return torch.nn.functional.softplus(raw_ds) + self.step_size_min

    @torch.no_grad()
    def predict_with_info(self, state_np: np.ndarray) -> tuple[float, dict]:
        self.eval()
        device = next(self.parameters()).device
        x = torch.as_tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
        ds = self.forward(x)
        ds_value = float(ds.squeeze(0).item())
        return ds_value, {}

    @torch.no_grad()
    def predict(self, state_np: np.ndarray) -> float:
        ds_value, _ = self.predict_with_info(state_np)
        return ds_value


def build_controller(controller_config: Dict[str, Any], input_dim: int = 8) -> NNController:
    return NNController(
        input_dim=int(controller_config.get("input_dim", input_dim)),
        hidden_dims=list(controller_config.get("hidden_dims", [64, 64])),
        dropout=float(controller_config.get("dropout", 0.1)),
        norm_type=str(controller_config.get("norm_type", "layernorm")),
        activation=str(controller_config.get("activation", "silu")),
        head_hidden_dim=controller_config.get("head_hidden_dim"),
        step_size_min=float(controller_config.get("step_size_min", 1e-6)),
        step_size_max=controller_config.get("step_size_max"),
    )


def build_controller_from_checkpoint(
    checkpoint: Dict[str, Any],
    controller_config: Dict[str, Any],
    input_dim: int = 8,
) -> NNController:
    merged_config = dict(controller_config)
    merged_config.update(checkpoint.get("model_config", {}))
    merged_config.setdefault("input_dim", input_dim)
    return build_controller(merged_config, input_dim=input_dim)
