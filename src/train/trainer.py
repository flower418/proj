from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.metrics import step_regression_metrics


class ControllerTrainer:
    def __init__(
        self,
        model,
        loss_fn,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        logger=None,
        scheduler=None,
        gradient_clip_norm: float | None = None,
    ):
        self.device = torch.device(device)
        self.model = model
        self.loss_fn = loss_fn.to(self.device) if hasattr(loss_fn, "to") else loss_fn
        self.optimizer = optimizer
        self.logger = logger
        self.scheduler = scheduler
        self.gradient_clip_norm = gradient_clip_norm
        self._validate_parameter_devices()

    def _is_on_target_device(self, tensor: torch.Tensor) -> bool:
        if tensor.device.type != self.device.type:
            return False
        if self.device.index is None:
            return True
        return tensor.device.index == self.device.index

    def _validate_parameter_devices(self) -> None:
        model_params = list(self.model.parameters())
        if model_params and any(not self._is_on_target_device(param) for param in model_params):
            raise ValueError(
                f"Model parameters must already be on {self.device} before creating ControllerTrainer."
            )
        optimizer_params = [param for group in self.optimizer.param_groups for param in group["params"]]
        if optimizer_params and any(not self._is_on_target_device(param) for param in optimizer_params):
            raise ValueError(
                f"Optimizer parameters must be created from model parameters on {self.device}."
            )

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        totals = {"loss": 0.0, "step_loss": 0.0}
        count = 0
        for batch in dataloader:
            x = batch["features"].to(self.device)
            ds_expert = batch["ds_expert"].to(self.device)
            ds_pred = self.model(x)
            loss, step_loss = self.loss_fn(ds_pred, ds_expert)
            self.optimizer.zero_grad()
            loss.backward()
            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            self.optimizer.step()
            totals["loss"] += float(loss.item())
            totals["step_loss"] += float(step_loss.item())
            count += 1
        if count == 0:
            return totals
        return {key: value / count for key, value in totals.items()}

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        totals = {"loss": 0.0, "step_loss": 0.0}
        count = 0
        all_ds_pred = []
        all_ds_true = []
        for batch in dataloader:
            x = batch["features"].to(self.device)
            ds_expert = batch["ds_expert"].to(self.device)
            ds_pred = self.model(x)
            loss, step_loss = self.loss_fn(ds_pred, ds_expert)
            totals["loss"] += float(loss.item())
            totals["step_loss"] += float(step_loss.item())
            all_ds_pred.append(ds_pred.detach().cpu().numpy().reshape(-1))
            all_ds_true.append(ds_expert.detach().cpu().numpy().reshape(-1))
            count += 1
        if count == 0:
            return totals
        metrics = {key: value / count for key, value in totals.items()}
        ds_pred = np.concatenate(all_ds_pred)
        ds_true = np.concatenate(all_ds_true)
        metrics.update(step_regression_metrics(ds_true, ds_pred))
        metrics["_raw"] = {"ds_pred": ds_pred, "ds_true": ds_true}
        return metrics

    def train(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        train_dataset=None,
        val_dataset=None,
        epochs: int = 50,
        early_stop_patience: int = 10,
        batch_size: int = 128,
        checkpoint_dir: Optional[str] = None,
    ):
        if train_loader is None:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if val_loader is None:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        history = []
        best_val = float("inf")
        best_state = None
        best_raw = None
        patience = 0
        checkpoint_path = None if checkpoint_dir is None else Path(checkpoint_dir)
        if checkpoint_path is not None:
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            raw = val_metrics.pop("_raw", None)
            history.append(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                    "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                }
            )
            if self.logger is not None:
                lr = float(self.optimizer.param_groups[0]["lr"])
                self.logger.log_model_weights(self.model, epoch)
                self.logger.log_epoch(epoch, train_metrics, val_metrics, lr)
                if raw is not None:
                    self.logger.log_prediction_scatter(raw["ds_pred"], raw["ds_true"], epoch)
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                best_state = copy.deepcopy(self.model.state_dict())
                if raw is not None:
                    best_raw = {key: np.copy(value) for key, value in raw.items()}
                patience = 0
                if checkpoint_path is not None:
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "model_config": self.model.get_config() if hasattr(self.model, "get_config") else None,
                            "epoch": epoch,
                            "val_loss": best_val,
                        },
                        checkpoint_path / "best_model.pt",
                    )
            else:
                patience += 1
            if patience >= early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch + 1}: "
                    f"validation loss did not improve for {early_stop_patience} epochs."
                )
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        if self.logger is not None:
            if best_raw is not None:
                self.logger.log_prediction_scatter(best_raw["ds_pred"], best_raw["ds_true"], len(history))
            self.logger.log_learning_curves()
        return history
