from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


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
            "val_step_loss": [],
            "learning_rate": [],
            "weight_norm": [],
            "grad_norm": [],
        }
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
        self.history["val_step_loss"].append(float(val_metrics.get("step_loss", 0.0)))
        self.history["learning_rate"].append(float(lr))
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
            f"lr={lr:.2e} "
            f"|W|={weight_norm:.3e} "
            f"|G|={grad_norm:.3e}"
        )

    def _save_final_summary(self) -> None:
        epochs = np.arange(1, len(self.history["train_loss"]) + 1)
        if len(epochs) == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        self._plot_log_loss(
            axes[0, 0],
            epochs,
            {
                "train": self.history["train_loss"],
                "val": self.history["val_loss"],
            },
            title="Total Loss (log scale)",
        )
        axes[0, 0].set_xlabel("Epoch")

        self._plot_log_loss(
            axes[0, 1],
            epochs,
            {
                "train_step": self.history["train_step_loss"],
                "val_step": self.history["val_step_loss"],
            },
            title="Step Loss (log scale)",
        )
        axes[0, 1].set_xlabel("Epoch")

        axes[1, 0].semilogy(epochs, np.maximum(self.history["learning_rate"], 1e-12), label="lr", linewidth=2)
        axes[1, 0].plot(epochs, np.maximum(self.history["weight_norm"], 1e-12), label="weight_norm", linewidth=2)
        axes[1, 0].plot(epochs, np.maximum(self.history["grad_norm"], 1e-12), label="grad_norm", linewidth=2)
        axes[1, 0].set_title("LR / Norm Trend")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        if self._latest_step_scatter is not None:
            ds_true, ds_pred = self._latest_step_scatter
            axes[1, 1].scatter(ds_true, ds_pred, alpha=0.35, s=12)
            lo = min(float(np.min(ds_true)), float(np.min(ds_pred)))
            hi = max(float(np.max(ds_true)), float(np.max(ds_pred)))
            axes[1, 1].plot([lo, hi], [lo, hi], "r--", linewidth=1.2)
            axes[1, 1].set_xscale("log")
            axes[1, 1].set_yscale("log")
            r2 = float(r2_score(ds_true, ds_pred)) if len(ds_true) > 1 else 0.0
            axes[1, 1].set_title(f"Final Step Scatter (R2={r2:.3f})")
            axes[1, 1].set_xlabel("True ds")
            axes[1, 1].set_ylabel("Pred ds")
            axes[1, 1].grid(True, alpha=0.3)
        elif self._initial_features is not None:
            axes[1, 1].hist(self._initial_features[:, 0], bins=40, alpha=0.8)
            axes[1, 1].set_title("Initial Feature f1 Distribution")
        else:
            axes[1, 1].axis("off")

        fig.suptitle("Training Summary", fontsize=16)
        fig.tight_layout()
        fig.savefig(self.summary_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def _plot_log_loss(
        self,
        ax,
        epochs: np.ndarray,
        series: Dict[str, list[float]],
        title: str,
    ) -> None:
        floor = 1e-12
        for label, values in series.items():
            arr = np.maximum(np.asarray(values, dtype=np.float64), floor)
            ax.semilogy(epochs, arr, label=label, linewidth=2)
        ax.set_title(title)
        ax.set_ylabel("Loss")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
