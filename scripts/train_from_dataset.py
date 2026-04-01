from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import _bootstrap  # noqa: F401

from src.data.dataset import create_dataloaders
from src.nn.controller import NNController
from src.nn.loss import ControllerLoss
from src.train.logger import TrainingLogger
from src.train.trainer import ControllerTrainer
from src.utils.config import load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser(description="从预生成数据集训练")
    parser.add_argument("--data-dir", type=str, required=True, help="数据集目录")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--experiment-name", type=str, default="model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint-dir", type=str, default="models")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    train_cfg = load_yaml_config(args.training_config, validate=False).get("defaults", {})

    # 加载数据
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # 创建模型
    model = NNController(
        input_dim=7,
        hidden_dims=config["controller"]["hidden_dims"],
        dropout=config["controller"]["dropout"],
        norm_type=config["controller"]["norm_type"],
        step_size_min=config["controller"]["step_size_min"],
        step_size_max=config["controller"]["step_size_max"],
    )

    # 损失函数
    loss_fn = ControllerLoss(
        lambda_step=config["training"]["lambda_step"],
        lambda_restart=config["training"]["lambda_restart"],
        alpha_restart=config["training"]["alpha_restart"],
        focal_gamma=config["training"]["focal_gamma"],
    )

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get("learning_rate", config["training"]["learning_rate"])
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_cfg.get("scheduler_factor", config["training"]["scheduler_factor"]),
        patience=train_cfg.get("scheduler_patience", config["training"]["scheduler_patience"]),
    )

    # 日志
    logger = TrainingLogger(log_dir="logs", experiment_name=args.experiment_name)
    logger.save_config({
        "config": config,
        "training_config": train_cfg,
        "args": vars(args),
    })
    sample_batch = next(iter(train_loader))
    logger.log_feature_distribution(sample_batch["features"].cpu().numpy(), epoch=0)

    # 训练器
    trainer = ControllerTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=args.device,
        logger=logger,
        scheduler=scheduler,
    )

    # 训练
    epochs = train_cfg.get("epochs", args.epochs)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stop_patience=train_cfg.get("early_stop_patience", 10),
        checkpoint_dir=args.checkpoint_dir,
    )

    # 保存训练历史
    history_path = Path(args.checkpoint_dir) / args.experiment_name / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w") as f:
        json.dump(history, f, indent=2)

    # 在测试集上评估
    print("\n在测试集上评估...")
    test_metrics = trainer.evaluate(test_loader)
    raw = test_metrics.pop("_raw", None)
    print(f"Test Loss: {test_metrics['loss']:.6f}")
    print(f"Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")
    print(f"Test F1: {test_metrics.get('f1', 0):.4f}")
    if raw is not None:
        logger.log_confusion_matrix(raw["y_true"], (raw["y_prob"] >= 0.5).astype(int), epoch=len(history))
        logger.log_prediction_scatter(raw["ds_pred"], raw["ds_true"], epoch=len(history))

    # 保存测试指标
    metrics_path = Path(args.checkpoint_dir) / args.experiment_name / "test_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(test_metrics, f, indent=2)

    logger.close()
    print(f"\n训练完成！epochs_run={len(history)} final_val={history[-1]['val']['loss']:.6f}")
    print(f"模型保存在：{args.checkpoint_dir}/{args.experiment_name}/best_model.pt")
    print(f"训练总图：{logger.summary_path}")


if __name__ == "__main__":
    main()
