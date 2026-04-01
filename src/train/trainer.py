from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


class ControllerTrainer:
    def __init__(self, model, loss_fn, optimizer: torch.optim.Optimizer, device: str = "cpu", logger=None, scheduler=None):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.scheduler = scheduler

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        totals = {"loss": 0.0, "step_loss": 0.0, "restart_loss": 0.0}
        count = 0
        for batch in dataloader:
            x = batch["features"].to(self.device)
            ds_expert = batch["ds_expert"].to(self.device)
            y_restart = batch["y_restart"].to(self.device)
            ds_pred, p_restart = self.model(x)
            loss, step_loss, restart_loss = self.loss_fn(ds_pred, ds_expert, p_restart, y_restart)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            totals["loss"] += float(loss.item())
            totals["step_loss"] += float(step_loss.item())
            totals["restart_loss"] += float(restart_loss.item())
            count += 1
        if count == 0:
            return totals
        return {key: value / count for key, value in totals.items()}

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        totals = {"loss": 0.0, "step_loss": 0.0, "restart_loss": 0.0}
        count = 0
        all_probs = []
        all_labels = []
        all_ds_pred = []
        all_ds_true = []
        for batch in dataloader:
            x = batch["features"].to(self.device)
            ds_expert = batch["ds_expert"].to(self.device)
            y_restart = batch["y_restart"].to(self.device)
            ds_pred, p_restart = self.model(x)
            loss, step_loss, restart_loss = self.loss_fn(ds_pred, ds_expert, p_restart, y_restart)
            totals["loss"] += float(loss.item())
            totals["step_loss"] += float(step_loss.item())
            totals["restart_loss"] += float(restart_loss.item())
            all_probs.append(p_restart.detach().cpu().numpy().reshape(-1))
            all_labels.append(y_restart.detach().cpu().numpy().reshape(-1))
            all_ds_pred.append(ds_pred.detach().cpu().numpy().reshape(-1))
            all_ds_true.append(ds_expert.detach().cpu().numpy().reshape(-1))
            count += 1
        if count == 0:
            return totals
        metrics = {key: value / count for key, value in totals.items()}
        y_prob = np.concatenate(all_probs)
        y_true = np.concatenate(all_labels).astype(np.int64)
        ds_pred = np.concatenate(all_ds_pred)
        ds_true = np.concatenate(all_ds_true)
        if self.logger is not None:
            metrics.update(self.logger.compute_classification_metrics(y_true, y_prob))
        metrics["_raw"] = {"y_true": y_true, "y_prob": y_prob, "ds_pred": ds_pred, "ds_true": ds_true}
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
        # 支持直接传入 DataLoader 或 Dataset
        if train_loader is None:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if val_loader is None:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        history = []
        best_val = float("inf")
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
                self.logger.log_epoch(epoch, train_metrics, val_metrics, lr)
                self.logger.log_model_weights(self.model, epoch)
                if raw is not None:
                    self.logger.log_confusion_matrix(raw["y_true"], (raw["y_prob"] >= 0.5).astype(np.int64), epoch)
                    self.logger.log_prediction_scatter(raw["ds_pred"], raw["ds_true"], epoch)
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                patience = 0
                if checkpoint_path is not None:
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "epoch": epoch,
                            "val_loss": best_val,
                        },
                        checkpoint_path / "best_model.pt",
                    )
            else:
                patience += 1
            if patience >= early_stop_patience:
                break
        if self.logger is not None:
            self.logger.log_learning_curves()
        return history
