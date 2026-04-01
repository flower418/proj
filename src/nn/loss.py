from __future__ import annotations

import torch
from torch import nn


class ControllerLoss(nn.Module):
    def __init__(
        self,
        lambda_step: float = 1.0,
        lambda_restart: float = 1.0,
        alpha_restart: float = 0.9,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.lambda_step = lambda_step
        self.lambda_restart = lambda_restart
        self.alpha = alpha_restart
        self.focal_gamma = focal_gamma

    def forward(self, ds_pred, ds_expert, p_restart, y_restart):
        ds_pred = ds_pred.reshape(-1)
        ds_expert = ds_expert.reshape(-1)
        p_restart = p_restart.reshape(-1)
        y_restart = y_restart.reshape(-1)

        step_mask = (y_restart < 0.5).to(ds_pred.dtype)
        if torch.sum(step_mask) > 0:
            log_ds_pred = torch.log(torch.clamp(ds_pred, min=1e-8))
            log_ds_expert = torch.log(torch.clamp(ds_expert, min=1e-8))
            step_loss = torch.sum(((log_ds_pred - log_ds_expert) ** 2) * step_mask) / torch.sum(step_mask)
        else:
            step_loss = torch.zeros((), dtype=ds_pred.dtype, device=ds_pred.device)

        eps = 1e-8
        bce = -(
            self.alpha * y_restart * torch.log(p_restart + eps)
            + (1 - self.alpha) * (1 - y_restart) * torch.log(1 - p_restart + eps)
        )
        pt = p_restart * y_restart + (1 - p_restart) * (1 - y_restart)
        restart_loss = torch.mean(bce * ((1 - pt) ** self.focal_gamma))
        total_loss = self.lambda_step * step_loss + self.lambda_restart * restart_loss
        return total_loss, step_loss, restart_loss
