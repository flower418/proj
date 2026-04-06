from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.linalg import svdvals


def plot_pseudospectrum_background(
    A: np.ndarray,
    epsilon: float,
    ax,
    resolution: int = 100,
    alpha: float = 0.18,
    padding: float = 0.2,
):
    """Draw a background contour for sigma_min(zI - A) = epsilon."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if xlim == (0.0, 1.0) and ylim == (0.0, 1.0):
        eigvals = np.linalg.eigvals(A)
        x_min = float(np.min(np.real(eigvals)))
        x_max = float(np.max(np.real(eigvals)))
        y_min = float(np.min(np.imag(eigvals)))
        y_max = float(np.max(np.imag(eigvals)))
        x_pad = max((x_max - x_min) * padding, 1.0)
        y_pad = max((y_max - y_min) * padding, 1.0)
        xlim = (x_min - x_pad, x_max + x_pad)
        ylim = (y_min - y_pad, y_max + y_pad)

    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], resolution),
        np.linspace(ylim[0], ylim[1], resolution),
    )
    zz = xx + 1j * yy
    n = A.shape[0]
    sigma_min = np.zeros_like(xx, dtype=np.float64)
    identity = np.eye(n, dtype=np.complex128)

    for i in range(resolution):
        for j in range(resolution):
            sigma_min[i, j] = float(np.min(svdvals(zz[i, j] * identity - A)))

    return ax.contour(
        xx,
        yy,
        sigma_min,
        levels=[epsilon],
        colors="0.55",
        linestyles="dashed",
        linewidths=1.0,
        alpha=alpha,
    )


def plot_trajectory(
    trajectory: np.ndarray,
    restart_indices: Optional[list] = None,
    step_sizes: Optional[np.ndarray] = None,
    A: Optional[np.ndarray] = None,
    epsilon: Optional[float] = None,
    ax=None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Visualize contour tracking with optional restart markers and step-size heatmap."""
    ax = ax or plt.gca()
    traj_array = np.asarray(trajectory, dtype=np.complex128)

    if A is not None and epsilon is not None:
        real_margin = max(np.ptp(np.real(traj_array)) * 0.2, 1.0)
        imag_margin = max(np.ptp(np.imag(traj_array)) * 0.2, 1.0)
        ax.set_xlim(np.real(traj_array).min() - real_margin, np.real(traj_array).max() + real_margin)
        ax.set_ylim(np.imag(traj_array).min() - imag_margin, np.imag(traj_array).max() + imag_margin)
        plot_pseudospectrum_background(A, epsilon, ax, resolution=100)

    if step_sizes is not None and len(step_sizes) == len(traj_array) - 1 and len(traj_array) >= 2:
        xy = np.column_stack([np.real(traj_array), np.imag(traj_array)])
        segments = np.stack([xy[:-1], xy[1:]], axis=1)
        step_array = np.asarray(step_sizes, dtype=np.float64)
        vmin = float(np.min(step_array))
        vmax = float(np.max(step_array))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-12
        line_collection = LineCollection(
            segments,
            cmap="viridis",
            norm=Normalize(vmin=vmin, vmax=vmax),
            linewidths=3.0,
            alpha=0.98,
            zorder=5,
        )
        line_collection.set_array(step_array)
        ax.add_collection(line_collection)
        ax.plot(
            np.real(traj_array),
            np.imag(traj_array),
            color="black",
            linewidth=0.8,
            alpha=0.25,
            zorder=4,
            label="Tracked Contour",
        )
        plt.colorbar(line_collection, ax=ax, label="Step size", fraction=0.046, pad=0.03, shrink=0.9)
    else:
        ax.plot(np.real(traj_array), np.imag(traj_array), color="tab:blue", linewidth=2.6, alpha=0.95, label="Tracked Contour", zorder=4)

    ax.scatter(np.real(traj_array[0]), np.imag(traj_array[0]), c="green", s=130, marker="o", label="Start", zorder=6, edgecolors="black", linewidths=0.6)
    ax.scatter(np.real(traj_array[-1]), np.imag(traj_array[-1]), c="red", s=110, marker="s", label="End", zorder=6, edgecolors="black", linewidths=0.6)

    if restart_indices:
        clipped = [idx for idx in restart_indices if 0 <= idx < len(traj_array)]
        if clipped:
            restart_points = traj_array[clipped]
            ax.scatter(
                np.real(restart_points),
                np.imag(restart_points),
                c="orange",
                s=110,
                marker="x",
                linewidths=2,
                label="SVD Restart",
                zorder=7,
            )

    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.set_title(
        title or (f"Pseudospectrum Contour (epsilon={epsilon})" if epsilon is not None else "Pseudospectrum Contour"),
        fontsize=11,
        pad=20,
        loc="left",
        wrap=True,
    )
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="lower right",
            frameon=True,
            fontsize=9,
            handlelength=1.8,
            columnspacing=1.2,
        )
    ax.set_aspect("equal")
    ax.figure.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(path, dpi=150, bbox_inches="tight")

    return ax


def plot_training_summary(history: list, save_path: str):
    """Save a compact 2x2 training summary figure."""
    epochs = range(len(history))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    train_loss = [h["train"]["loss"] for h in history]
    val_loss = [h["val"]["loss"] for h in history]
    axes[0, 0].plot(epochs, train_loss, "b-", label="Train", linewidth=2)
    axes[0, 0].plot(epochs, val_loss, "r-", label="Val", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    train_step = [h["train"]["step_loss"] for h in history]
    train_restart = [h["train"]["restart_loss"] for h in history]
    axes[0, 1].plot(epochs, train_step, "g-", label="Step Loss", linewidth=2)
    axes[0, 1].plot(epochs, train_restart, "m-", label="Restart Loss", linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Component Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    if history and "accuracy" in history[0]["val"]:
        val_acc = [h["val"]["accuracy"] for h in history]
        val_f1 = [h["val"]["f1"] for h in history]
        axes[1, 0].plot(epochs, val_acc, "b-o", label="Accuracy", linewidth=2)
        axes[1, 0].plot(epochs, val_f1, "r-s", label="F1 Score", linewidth=2)
        axes[1, 0].legend()
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_title("Validation Metrics")
    axes[1, 0].grid(True, alpha=0.3)

    learning_rates = [h.get("learning_rate", 1e-3) for h in history]
    axes[1, 1].semilogy(epochs, learning_rates, "k-", linewidth=2)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].set_title("Learning Rate Schedule")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
