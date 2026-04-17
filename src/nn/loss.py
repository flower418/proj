from __future__ import annotations

import torch
from torch import nn


class ControllerLoss(nn.Module):
    def __init__(self, lambda_step: float = 1.0):
        super().__init__()
        self.lambda_step = float(lambda_step)

    def forward(self, ds_pred, ds_expert):
        ds_pred = ds_pred.reshape(-1)
        ds_expert = ds_expert.reshape(-1)

        log_ds_pred = torch.log(torch.clamp(ds_pred, min=1e-8))
        log_ds_expert = torch.log(torch.clamp(ds_expert, min=1e-8))
        step_loss = torch.mean((log_ds_pred - log_ds_expert) ** 2)
        total_loss = self.lambda_step * step_loss
        return total_loss, step_loss
