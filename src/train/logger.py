from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, r2_score


class TrainingLogger:
    """Local logger that prints every epoch and saves one final summary figure."""

    def __init__(self, log_dir: str = "logs", experiment_name: Optional[str] = None):
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.log_dir / "training_summary.png"
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
        self._latest_confusion_matrix: np.ndarray | None = None
        self._latest_step_scatter: tuple[np.ndarray, np.ndarray] | None = None
        self._initial_features: np.ndarray | None = None

    def save_config(self, config: Dict) -> None:
        with (self.log_dir / "config.json").open("w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, ensure_ascii=False)

    def log_scalars(self, scalars: Dict[str, float], global_step: int, prefix: str = "") -> None:
        payload = {"step": global_step, "prefix": prefix, "scalars": {k: float(v) for k, v in scalars.items()}}
        with (self.log_dir / "scalars.jsonl").open("a", encoding="utf-8") as fh:
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
        self._print_epoch_summary(epoch, train_metrics, val_metrics, lr)

    def log_model_weights(self, model, epoch: int) -> None:
        weight_norms = []
        grad_norms = []
        for _, param in model.named_parameters():
            weight_norms.append(float(np.linalg.norm(param.detach().cpu().numpy().ravel())))
            if param.grad is not None:
                grad_norms.append(float(np.linalg.norm(param.grad.detach().cpu().numpy().ravel())))

        self.history["weight_norm"].append(float(np.mean(weight_norms) if weight_norms else 0.0))
        self.history["grad_norm"].append(float(np.mean(grad_norms) if grad_norms else 0.0))
        payload = {
            "epoch": epoch,
            "mean_weight_norm": self.history["weight_norm"][-1],
            "mean_grad_norm": self.history["grad_norm"][-1],
        }
        with (self.log_dir / "model_norms.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, epoch: int) -> None:
        del epoch
        self._latest_confusion_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    def log_prediction_scatter(self, ds_pred: np.ndarray, ds_true: np.ndarray, epoch: int) -> None:
        del epoch
        mask = (ds_true > 0.0) & (ds_pred > 0.0)
        if not np.any(mask):
            self._latest_step_scatter = None
            return
        self._latest_step_scatter = (
            np.asarray(ds_true[mask], dtype=np.float64),
            np.asarray(ds_pred[mask], dtype=np.float64),
        )

    def log_feature_distribution(self, features: np.ndarray, epoch: int) -> None:
        del epoch
        self._initial_features = np.asarray(features, dtype=np.float32)

    def log_learning_curves(self) -> None:
        self._save_final_summary()

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
        self._save_final_summary()

    def _save_history(self) -> None:
        with (self.log_dir / "history.json").open("w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2, ensure_ascii=False)

    def _print_epoch_summary(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float], lr: float) -> None:
        weight_norm = self.history["weight_norm"][-1] if self.history["weight_norm"] else 0.0
        grad_norm = self.history["grad_norm"][-1] if self.history["grad_norm"] else 0.0
        print(
            f"[Epoch {epoch + 1:03d}] "
            f"train_loss={train_metrics.get('loss', 0.0):.6f} "
            f"val_loss={val_metrics.get('loss', 0.0):.6f} "
            f"step={val_metrics.get('step_loss', 0.0):.6f} "
            f"restart={val_metrics.get('restart_loss', 0.0):.6f} "
            f"acc={val_metrics.get('accuracy', 0.0):.4f} "
            f"f1={val_metrics.get('f1', 0.0):.4f} "
            f"lr={lr:.2e} "
            f"|W|={weight_norm:.3e} "
            f"|G|={grad_norm:.3e}"
        )

    def _save_final_summary(self) -> None:
        epochs = np.arange(1, len(self.history["train_loss"]) + 1)
        if len(epochs) == 0:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        axes[0, 0].plot(epochs, self.history["train_loss"], label="train", linewidth=2)
        axes[0, 0].plot(epochs, self.history["val_loss"], label="val", linewidth=2)
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        axes[0, 1].plot(epochs, self.history["train_step_loss"], label="train_step", linewidth=2)
        axes[0, 1].plot(epochs, self.history["val_step_loss"], label="val_step", linewidth=2)
        axes[0, 1].plot(epochs, self.history["train_restart_loss"], label="train_restart", linewidth=2)
        axes[0, 1].plot(epochs, self.history["val_restart_loss"], label="val_restart", linewidth=2)
        axes[0, 1].set_title("Component Losses")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        axes[0, 2].plot(epochs, self.history["val_accuracy"], label="accuracy", linewidth=2)
        axes[0, 2].plot(epochs, self.history["val_f1"], label="f1", linewidth=2)
        axes[0, 2].plot(epochs, self.history["val_precision"], label="precision", linewidth=1.5)
        axes[0, 2].plot(epochs, self.history["val_recall"], label="recall", linewidth=1.5)
        axes[0, 2].set_ylim(0.0, 1.05)
        axes[0, 2].set_title("Validation Metrics")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()

        axes[1, 0].semilogy(epochs, np.maximum(self.history["learning_rate"], 1e-12), label="lr", linewidth=2)
        axes[1, 0].plot(epochs, np.maximum(self.history["weight_norm"], 1e-12), label="weight_norm", linewidth=2)
        axes[1, 0].plot(epochs, np.maximum(self.history["grad_norm"], 1e-12), label="grad_norm", linewidth=2)
        axes[1, 0].set_title("LR / Norm Trend")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        if self._latest_confusion_matrix is not None:
            cm = self._latest_confusion_matrix
            im = axes[1, 1].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
            axes[1, 1].set_title("Final Confusion Matrix")
            axes[1, 1].set_xticks([0, 1])
            axes[1, 1].set_yticks([0, 1])
            axes[1, 1].set_xticklabels(["keep", "restart"])
            axes[1, 1].set_yticklabels(["keep", "restart"])
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axes[1, 1].text(j, i, str(cm[i, j]), ha="center", va="center")
        else:
            axes[1, 1].axis("off")

        if self._latest_step_scatter is not None:
            ds_true, ds_pred = self._latest_step_scatter
            axes[1, 2].scatter(ds_true, ds_pred, alpha=0.35, s=12)
            lo = min(float(np.min(ds_true)), float(np.min(ds_pred)))
            hi = max(float(np.max(ds_true)), float(np.max(ds_pred)))
            axes[1, 2].plot([lo, hi], [lo, hi], "r--", linewidth=1.2)
            axes[1, 2].set_xscale("log")
            axes[1, 2].set_yscale("log")
            r2 = float(r2_score(ds_true, ds_pred)) if len(ds_true) > 1 else 0.0
            axes[1, 2].set_title(f"Final Step Scatter (R2={r2:.3f})")
            axes[1, 2].set_xlabel("True ds")
            axes[1, 2].set_ylabel("Pred ds")
            axes[1, 2].grid(True, alpha=0.3)
        elif self._initial_features is not None:
            axes[1, 2].hist(self._initial_features[:, 0], bins=40, alpha=0.8)
            axes[1, 2].set_title("Initial Feature f1 Distribution")
        else:
            axes[1, 2].axis("off")

        fig.suptitle("Training Summary", fontsize=16)
        fig.tight_layout()
        fig.savefig(self.summary_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
