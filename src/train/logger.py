from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, r2_score
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """TensorBoard-backed logger for controller training."""

    def __init__(self, log_dir: str = "logs", experiment_name: Optional[str] = None):
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_step_loss": [],
            "train_restart_loss": [],
            "val_step_loss": [],
            "val_restart_loss": [],
            "learning_rate": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
        }

    def save_config(self, config: Dict) -> None:
        config_path = self.log_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, ensure_ascii=False)

    def log_scalars(self, scalars: Dict[str, float], global_step: int, prefix: str = "") -> None:
        for name, value in scalars.items():
            tag = f"{prefix}{name}" if prefix else name
            self.writer.add_scalar(tag, value, global_step)

    def log_epoch(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float], lr: float) -> None:
        self.history["train_loss"].append(train_metrics.get("loss", 0.0))
        self.history["val_loss"].append(val_metrics.get("loss", 0.0))
        self.history["train_step_loss"].append(train_metrics.get("step_loss", 0.0))
        self.history["train_restart_loss"].append(train_metrics.get("restart_loss", 0.0))
        self.history["val_step_loss"].append(val_metrics.get("step_loss", 0.0))
        self.history["val_restart_loss"].append(val_metrics.get("restart_loss", 0.0))
        self.history["learning_rate"].append(lr)
        self.history["val_accuracy"].append(val_metrics.get("accuracy", 0.0))
        self.history["val_precision"].append(val_metrics.get("precision", 0.0))
        self.history["val_recall"].append(val_metrics.get("recall", 0.0))
        self.history["val_f1"].append(val_metrics.get("f1", 0.0))
        self.log_scalars(train_metrics, epoch, prefix="train/")
        self.log_scalars(val_metrics, epoch, prefix="val/")
        self.writer.add_scalar("train/learning_rate", lr, epoch)

    def log_model_weights(self, model, epoch: int) -> None:
        for name, param in model.named_parameters():
            self.writer.add_histogram(f"weights/{name}", param.detach().cpu().numpy(), epoch)
            if param.grad is not None:
                self.writer.add_histogram(f"grads/{name}", param.grad.detach().cpu().numpy(), epoch)

    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, epoch: int, class_names: Optional[Iterable[str]] = None) -> None:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax)
        if class_names is None:
            class_names = ["keep", "restart"]
        class_names = list(class_names)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.tight_layout()
        self.writer.add_figure("val/confusion_matrix", fig, epoch)
        plt.close(fig)

    def log_prediction_scatter(self, ds_pred: np.ndarray, ds_true: np.ndarray, epoch: int) -> None:
        mask = (ds_true > 0.0) & (ds_pred > 0.0)
        if not np.any(mask):
            return
        ds_true = ds_true[mask]
        ds_pred = ds_pred[mask]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(ds_true, ds_pred, alpha=0.3, s=12)
        lo = min(float(np.min(ds_true)), float(np.min(ds_pred)))
        hi = max(float(np.max(ds_true)), float(np.max(ds_pred)))
        ax.plot([lo, hi], [lo, hi], "r--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("True ds")
        ax.set_ylabel("Predicted ds")
        ax.set_title("Step Size Prediction")
        fig.tight_layout()
        self.writer.add_figure("val/step_size_scatter", fig, epoch)
        if len(ds_true) > 1:
            self.writer.add_scalar("val/step_size_r2", r2_score(ds_true, ds_pred), epoch)
        plt.close(fig)

    def log_feature_distribution(self, features: np.ndarray, epoch: int) -> None:
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        axes = axes.flatten()
        feature_names = ["f1", "f2", "f3", "f4", "f5", "f6", "f7"]
        for idx, name in enumerate(feature_names):
            axes[idx].hist(features[:, idx], bins=40, alpha=0.75)
            axes[idx].set_title(name)
        axes[-1].axis("off")
        fig.tight_layout()
        self.writer.add_figure("train/feature_distribution", fig, epoch)
        plt.close(fig)

    def log_learning_curves(self) -> None:
        epochs = np.arange(1, len(self.history["train_loss"]) + 1)
        if len(epochs) == 0:
            return
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(epochs, self.history["train_loss"], label="train")
        axes[0, 0].plot(epochs, self.history["val_loss"], label="val")
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].legend()
        axes[0, 1].plot(epochs, self.history["train_step_loss"], label="train_step")
        axes[0, 1].plot(epochs, self.history["val_step_loss"], label="val_step")
        axes[0, 1].set_title("Step Loss")
        axes[0, 1].legend()
        axes[1, 0].plot(epochs, self.history["train_restart_loss"], label="train_restart")
        axes[1, 0].plot(epochs, self.history["val_restart_loss"], label="val_restart")
        axes[1, 0].set_title("Restart Loss")
        axes[1, 0].legend()
        axes[1, 1].plot(epochs, self.history["val_accuracy"], label="acc")
        axes[1, 1].plot(epochs, self.history["val_f1"], label="f1")
        axes[1, 1].set_title("Validation Metrics")
        axes[1, 1].legend()
        fig.tight_layout()
        fig.savefig(self.log_dir / "learning_curves.png", dpi=160)
        plt.close(fig)

    def compute_classification_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        y_pred = (y_prob >= 0.5).astype(np.int64)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
