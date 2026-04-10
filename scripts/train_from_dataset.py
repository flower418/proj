from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import _bootstrap  # noqa: F401

from src.data.dataset import create_dataloaders
from src.nn.controller import build_controller
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
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="models")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--norm-type", choices=["layernorm", "batchnorm"], default=None)
    parser.add_argument("--activation", choices=["relu", "gelu", "silu"], default=None)
    parser.add_argument("--head-hidden-dim", type=int, default=None)
    parser.add_argument("--step-size-min", type=float, default=None)
    parser.add_argument("--step-size-max", type=float, default=None)
    parser.add_argument("--lambda-step", type=float, default=None)
    parser.add_argument("--lambda-restart", type=float, default=None)
    parser.add_argument("--alpha-restart", type=float, default=None)
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    train_cfg = load_yaml_config(args.training_config, validate=False).get("defaults", {})
    controller_cfg = dict(config["controller"])
    if args.hidden_dims is not None:
        controller_cfg["hidden_dims"] = args.hidden_dims
    if args.dropout is not None:
        controller_cfg["dropout"] = args.dropout
    if args.norm_type is not None:
        controller_cfg["norm_type"] = args.norm_type
    if args.activation is not None:
        controller_cfg["activation"] = args.activation
    if args.head_hidden_dim is not None:
        controller_cfg["head_hidden_dim"] = args.head_hidden_dim
    if args.step_size_min is not None:
        controller_cfg["step_size_min"] = args.step_size_min
    if args.step_size_max is not None:
        controller_cfg["step_size_max"] = args.step_size_max

    training_cfg = dict(config["training"])
    if args.lambda_step is not None:
        training_cfg["lambda_step"] = args.lambda_step
    if args.lambda_restart is not None:
        training_cfg["lambda_restart"] = args.lambda_restart
    if args.alpha_restart is not None:
        training_cfg["alpha_restart"] = args.alpha_restart
    if args.focal_gamma is not None:
        training_cfg["focal_gamma"] = args.focal_gamma
    if args.gradient_clip_norm is not None:
        training_cfg["gradient_clip_norm"] = args.gradient_clip_norm
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", config["training"]["batch_size"])
    learning_rate = args.learning_rate if args.learning_rate is not None else train_cfg.get("learning_rate", training_cfg["learning_rate"])
    weight_decay = args.weight_decay if args.weight_decay is not None else train_cfg.get("weight_decay", training_cfg.get("weight_decay", 0.0))
    early_stop_patience = args.early_stop_patience if args.early_stop_patience is not None else train_cfg.get("early_stop_patience", training_cfg.get("early_stop_patience", 10))
    gradient_clip_norm = training_cfg.get("gradient_clip_norm")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA 训练，但当前 PyTorch 未检测到可用 GPU。")

    # 加载数据
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir,
        batch_size=batch_size,
        num_workers=args.num_workers,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # 创建模型
    model = build_controller(
        controller_cfg,
        input_dim=int(controller_cfg.get("input_dim", config["controller"].get("input_dim", 7))),
    ).to(device)

    # 损失函数
    loss_fn = ControllerLoss(
        lambda_step=training_cfg["lambda_step"],
        lambda_restart=training_cfg["lambda_restart"],
        alpha_restart=training_cfg["alpha_restart"],
        focal_gamma=training_cfg["focal_gamma"],
    )

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_cfg.get("scheduler_factor", config["training"]["scheduler_factor"]),
        patience=train_cfg.get("scheduler_patience", config["training"]["scheduler_patience"]),
    )

    # 日志
    logger = TrainingLogger(log_dir=args.log_dir, experiment_name=args.experiment_name)
    logger.save_config({
        "config": config,
        "training_config": train_cfg,
        "effective_controller_config": controller_cfg,
        "effective_training_config": {
            **training_cfg,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "early_stop_patience": early_stop_patience,
        },
        "args": vars(args),
    })
    if len(train_loader) > 0:
        sample_batch = next(iter(train_loader))
        logger.log_feature_distribution(sample_batch["features"].cpu().numpy(), epoch=0)

    # 训练器
    trainer = ControllerTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        logger=logger,
        scheduler=scheduler,
        gradient_clip_norm=gradient_clip_norm,
    )

    # 训练
    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", config["training"]["epochs"])
    experiment_dir = Path(args.checkpoint_dir) / args.experiment_name
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stop_patience=early_stop_patience,
        checkpoint_dir=str(experiment_dir),
    )
    if not history:
        raise RuntimeError("训练未运行任何 epoch，请检查 epochs 和数据集划分。")

    # 保存训练历史
    history_path = experiment_dir / "training_history.json"
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
    metrics_path = experiment_dir / "test_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(test_metrics, f, indent=2)

    logger.close()
    print(f"\n训练完成！epochs_run={len(history)} final_val={history[-1]['val']['loss']:.6f}")
    print(f"模型保存在：{experiment_dir / 'best_model.pt'}")
    print(f"训练总图：{logger.summary_path}")


if __name__ == "__main__":
    main()
