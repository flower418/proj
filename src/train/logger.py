from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, r2_score


class TrainingLogger:
    """Local-file training logger with progressive visual summaries."""

    def __init__(self, log_dir: str = "logs", experiment_name: Optional[str] = None):
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir) / experiment_name
        self.fig_dir = self.log_dir / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
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
            "weight_norm": [],
            "grad_norm": [],
        }

    def save_config(self, config: Dict) -> None:
        config_path = self.log_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, ensure_ascii=False)

    def log_scalars(self, scalars: Dict[str, float], global_step: int, prefix: str = "") -> None:
        scalars_path = self.log_dir / "scalars.jsonl"
        payload = {"step": global_step, "prefix": prefix, "scalars": {k: float(v) for k, v in scalars.items()}}
        with scalars_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log_epoch(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float], lr: float) -> None:
        self.history["train_loss"].append(float(train_metrics.get("loss", 0.0)))
        self.history["val_loss"].append(float(val_metrics.get("loss", 0.0)))
        self.history["train_step_loss"].append(float(train_metrics.get("step_loss", 0.0)))
        self.history["train_restart_loss"].append(float(train_metrics.get("restart_loss", 0.0)))
        self.history["val_step_loss"].append(float(val_metrics.get("step_loss", 0.0)))
        self.history["val_restart_loss"].append(float(val_metrics.get("restart_loss", 0.0)))
        self.history["learning_rate"].append(float(lr))
        self.history["val_accuracy"].append(float(val_metrics.get("accuracy", 0.0)))
        self.history["val_precision"].append(float(val_metrics.get("precision", 0.0)))
        self.history["val_recall"].append(float(val_metrics.get("recall", 0.0)))
        self.history["val_f1"].append(float(val_metrics.get("f1", 0.0)))
        self.log_scalars(train_metrics, epoch, prefix="train/")
        self.log_scalars(val_metrics, epoch, prefix="val/")
        self._save_history()
        self._save_epoch_dashboard(epoch)

    def log_model_weights(self, model, epoch: int) -> None:
        weight_norms = {}
        grad_norms = {}
        for name, param in model.named_parameters():
            weight_norms[name] = float(np.linalg.norm(param.detach().cpu().numpy().ravel()))
            if param.grad is not None:
                grad_norms[name] = float(np.linalg.norm(param.grad.detach().cpu().numpy().ravel()))

        self.history["weight_norm"].append(float(np.mean(list(weight_norms.values())) if weight_norms else 0.0))
        self.history["grad_norm"].append(float(np.mean(list(grad_norms.values())) if grad_norms else 0.0))

        payload = {"epoch": epoch, "weight_norms": weight_norms, "grad_norms": grad_norms}
        with (self.log_dir / "model_norms.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        if weight_norms:
            names = list(weight_norms.keys())
            axes[0].bar(range(len(names)), list(weight_norms.values()))
            axes[0].set_title("Weight Norms")
            axes[0].set_xticks(range(len(names)))
            axes[0].set_xticklabels(names, rotation=75, fontsize=7)
        if grad_norms:
            names = list(grad_norms.keys())
            axes[1].bar(range(len(names)), list(grad_norms.values()), color="tab:orange")
            axes[1].set_title("Gradient Norms")
            axes[1].set_xticks(range(len(names)))
            axes[1].set_xticklabels(names, rotation=75, fontsize=7)
        fig.tight_layout()
        fig.savefig(self.fig_dir / f"parameter_norms_epoch_{epoch:04d}.png", dpi=150, bbox_inches="tight")
        fig.savefig(self.fig_dir / "parameter_norms_latest.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        self._save_history()

    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, epoch: int, class_names: Optional[Iterable[str]] = None) -> None:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax)
        class_names = list(class_names or ["keep", "restart"])
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix Epoch {epoch}")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(self.fig_dir / f"confusion_matrix_epoch_{epoch:04d}.png", dpi=150, bbox_inches="tight")
        fig.savefig(self.fig_dir / "confusion_matrix_latest.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def log_prediction_scatter(self, ds_pred: np.ndarray, ds_true: np.ndarray, epoch: int) -> None:
        mask = (ds_true > 0.0) & (ds_pred > 0.0)
        if not np.any(mask):
            return
        ds_true = ds_true[mask]
        ds_pred = ds_pred[mask]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(ds_true, ds_pred, alpha=0.35, s=14)
        lo = min(float(np.min(ds_true)), float(np.min(ds_pred)))
        hi = max(float(np.max(ds_true)), float(np.max(ds_pred)))
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("True Step Size")
        ax.set_ylabel("Predicted Step Size")
        r2 = float(r2_score(ds_true, ds_pred)) if len(ds_true) > 1 else 0.0
        ax.set_title(f"Step Prediction Epoch {epoch} | R2={r2:.3f}")
        fig.tight_layout()
        fig.savefig(self.fig_dir / f"step_scatter_epoch_{epoch:04d}.png", dpi=150, bbox_inches="tight")
        fig.savefig(self.fig_dir / "step_scatter_latest.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def log_feature_distribution(self, features: np.ndarray, epoch: int) -> None:
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        axes = axes.flatten()
        feature_names = ["f1", "f2", "f3", "f4", "f5", "f6", "f7"]
        for idx, name in enumerate(feature_names):
            axes[idx].hist(features[:, idx], bins=40, alpha=0.75, color="tab:blue")
            axes[idx].set_title(name)
        axes[-1].axis("off")
        fig.tight_layout()
        fig.savefig(self.fig_dir / f"feature_distribution_epoch_{epoch:04d}.png", dpi=150, bbox_inches="tight")
        fig.savefig(self.fig_dir / "feature_distribution_latest.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def log_learning_curves(self) -> None:
        self._save_epoch_dashboard(len(self.history["train_loss"]) - 1)

    def compute_classification_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        y_pred = (y_prob >= 0.5).astype(np.int64)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

    def close(self) -> None:
        self._save_history()

    def _save_history(self) -> None:
        with (self.log_dir / "history.json").open("w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2, ensure_ascii=False)

    def _save_epoch_dashboard(self, epoch: int) -> None:
        epochs = np.arange(1, len(self.history["train_loss"]) + 1)
        if len(epochs) == 0:
            return

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        axes[0, 0].plot(epochs, self.history["train_loss"], label="train", linewidth=2)
        axes[0, 0].plot(epochs, self.history["val_loss"], label="val", linewidth=2)
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        axes[0, 1].plot(epochs, self.history["train_step_loss"], label="train_step", linewidth=2)
        axes[0, 1].plot(epochs, self.history["val_step_loss"], label="val_step", linewidth=2)
        axes[0, 1].set_title("Step Regression Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        axes[0, 2].plot(epochs, self.history["train_restart_loss"], label="train_restart", linewidth=2)
        axes[0, 2].plot(epochs, self.history["val_restart_loss"], label="val_restart", linewidth=2)
        axes[0, 2].set_title("Restart Classification Loss")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()

        axes[1, 0].plot(epochs, self.history["val_accuracy"], label="accuracy", linewidth=2)
        axes[1, 0].plot(epochs, self.history["val_f1"], label="f1", linewidth=2)
        axes[1, 0].plot(epochs, self.history["val_precision"], label="precision", linewidth=1.5, alpha=0.9)
        axes[1, 0].plot(epochs, self.history["val_recall"], label="recall", linewidth=1.5, alpha=0.9)
        axes[1, 0].set_title("Validation Classification Metrics")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylim(0.0, 1.05)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        axes[1, 1].semilogy(epochs, np.maximum(self.history["learning_rate"], 1e-12), label="lr", linewidth=2)
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        axes[1, 2].plot(epochs[: len(self.history["weight_norm"])], self.history["weight_norm"], label="weight_norm", linewidth=2)
        axes[1, 2].plot(epochs[: len(self.history["grad_norm"])], self.history["grad_norm"], label="grad_norm", linewidth=2)
        axes[1, 2].set_title("Parameter / Gradient Norm Trend")
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend()

        fig.suptitle(f"Training Overview Through Epoch {epoch}", fontsize=14)
        fig.tight_layout()
        fig.savefig(self.fig_dir / f"training_dashboard_epoch_{epoch:04d}.png", dpi=160, bbox_inches="tight")
        fig.savefig(self.fig_dir / "training_dashboard_latest.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
