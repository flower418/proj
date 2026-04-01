from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze saved training history.")
    parser.add_argument("--history", default="models/training_history.json")
    parser.add_argument("--output", default="logs/training_summary.png")
    return parser.parse_args()


def main():
    args = parse_args()
    history_path = Path(args.history)
    with history_path.open("r", encoding="utf-8") as fh:
        history = json.load(fh)
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train"]["loss"] for item in history]
    val_loss = [item["val"]["loss"] for item in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="train")
    ax.plot(epochs, val_loss, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.set_title("Training Summary")
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"saved analysis to {out}")


if __name__ == "__main__":
    main()
