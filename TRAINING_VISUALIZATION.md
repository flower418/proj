# 训练过程可视化指南

## 1. 概述

本指南介绍如何实时监控和可视化训练过程，包括：
- Loss 曲线（总 Loss、步长 Loss、重启 Loss）
- 学习率变化
- 验证集指标
- 预测 vs 真实值对比
- 混淆矩阵（重启决策）
- 特征分布演化

---

## 2. TensorBoard 集成

### 2.1 训练日志记录器

```python
# src/train/logger.py

import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import json

class TrainingLogger:
    """
    训练日志记录器 - 支持 TensorBoard 和本地保存
    """
    
    def __init__(self, log_dir: str = 'logs/', 
                 experiment_name: str = None):
        """
        :param log_dir: 日志保存目录
        :param experiment_name: 实验名称 (用于子目录)
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard Writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # 本地历史记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_step_loss': [],
            'train_restart_loss': [],
            'val_step_loss': [],
            'val_restart_loss': [],
            'learning_rate': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }
        
        # 保存配置
        self.config_path = self.log_dir / 'config.json'
        
    def log_scalars(self, scalars: dict, global_step: int, 
                    prefix: str = ''):
        """
        记录标量值到 TensorBoard
        
        :param scalars: 字典 {name: value}
        :param global_step: 全局步数
        :param prefix: 名称前缀
        """
        for name, value in scalars.items():
            full_name = f"{prefix}{name}" if prefix else name
            self.writer.add_scalar(full_name, value, global_step)
    
    def log_epoch(self, epoch: int, train_metrics: dict, 
                  val_metrics: dict, lr: float):
        """
        记录一个 epoch 的完整指标
        
        :param epoch: 当前 epoch
        :param train_metrics: 训练集指标
        :param val_metrics: 验证集指标
        :param lr: 当前学习率
        """
        # 更新历史
        self.history['train_loss'].append(train_metrics.get('total_loss', 0))
        self.history['val_loss'].append(val_metrics.get('total_loss', 0))
        self.history['train_step_loss'].append(train_metrics.get('step_loss', 0))
        self.history['train_restart_loss'].append(train_metrics.get('restart_loss', 0))
        self.history['learning_rate'].append(lr)
        
        if val_metrics:
            self.history['val_accuracy'].append(val_metrics.get('accuracy', 0))
            self.history['val_precision'].append(val_metrics.get('precision', 0))
            self.history['val_recall'].append(val_metrics.get('recall', 0))
            self.history['val_f1'].append(val_metrics.get('f1', 0))
        
        # 记录到 TensorBoard
        self.log_scalars(train_metrics, epoch, prefix='train/')
        self.log_scalars(val_metrics, epoch, prefix='val/')
        self.writer.add_scalar('train/learning_rate', lr, epoch)
        
        # 打印进度
        self._print_epoch_summary(epoch, train_metrics, val_metrics, lr)
    
    def _print_epoch_summary(self, epoch: int, train_metrics: dict,
                             val_metrics: dict, lr: float):
        """打印 epoch 摘要"""
        print(f"\n{'='*60}")
        print(f"Epoch {epoch:03d} | LR: {lr:.2e}")
        print(f"{'-'*60}")
        print(f"Train Loss: {train_metrics.get('total_loss', 0):.6f} | "
              f"Step: {train_metrics.get('step_loss', 0):.6f} | "
              f"Restart: {train_metrics.get('restart_loss', 0):.6f}")
        if val_metrics:
            print(f"Val   Loss: {val_metrics.get('total_loss', 0):.6f} | "
                  f"Acc: {val_metrics.get('accuracy', 0):.4f} | "
                  f"F1: {val_metrics.get('f1', 0):.4f}")
        print(f"{'='*60}\n")
    
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             epoch: int, class_names: list = None):
        """
        记录混淆矩阵
        
        :param y_true: 真实标签
        :param y_pred: 预测标签
        :param epoch: 当前 epoch
        :param class_names: 类别名称列表
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        # 绘制热力图
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        if class_names:
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)
        
        # 添加数值标注
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        
        # 记录到 TensorBoard
        self.writer.add_figure('confusion_matrix', plt.gcf(), epoch)
        plt.close()
    
    def log_prediction_scatter(self, ds_pred: np.ndarray, ds_true: np.ndarray,
                               epoch: int, max_points: int = 500):
        """
        记录步长预测 vs 真实值散点图
        
        :param ds_pred: 预测步长
        :param ds_true: 真实步长
        :param epoch: 当前 epoch
        :param max_points: 最多显示的点数
        """
        # 采样
        if len(ds_pred) > max_points:
            indices = np.random.choice(len(ds_pred), max_points, replace=False)
            ds_pred = ds_pred[indices]
            ds_true = ds_true[indices]
        
        plt.figure(figsize=(8, 8))
        plt.scatter(ds_true, ds_pred, alpha=0.3, s=10)
        
        # 理想对角线
        min_val = min(ds_true.min(), ds_pred.min())
        max_val = max(ds_true.max(), ds_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
        
        plt.xlabel('True Step Size (ds)')
        plt.ylabel('Predicted Step Size (ds)')
        plt.title(f'Step Size Prediction (Epoch {epoch})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        
        self.writer.add_figure('step_size_scatter', plt.gcf(), epoch)
        plt.close()
        
        # 计算并记录 R²
        from sklearn.metrics import r2_score
        r2 = r2_score(ds_true, ds_pred)
        self.writer.add_scalar('step_size/R2_score', r2, epoch)
    
    def log_feature_distribution(self, features: np.ndarray, epoch: int,
                                 feature_names: list = None):
        """
        记录特征分布直方图
        
        :param features: 特征矩阵 [N, 7]
        :param epoch: 当前 epoch
        :param feature_names: 特征名称列表
        """
        if feature_names is None:
            feature_names = ['f1:σ误差', 'f2:u 漂移', 'f3:v 漂移', 
                            'f4:残差', 'f5:梯度模', 'f6:曲率', 'f7:迭代数']
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i in range(features.shape[1]):
            axes[i].hist(features[:, i], bins=50, alpha=0.7, log=True)
            axes[i].set_xlabel(feature_names[i])
            axes[i].set_ylabel('Count')
            axes[i].set_title(f'{feature_names[i]} 分布')
            axes[i].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        axes[-1].axis('off')
        
        plt.tight_layout()
        self.writer.add_figure('feature_distribution', fig, epoch)
        plt.close()
    
    def log_learning_curves(self):
        """
        生成并保存学习曲线图像
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # 1. 总 Loss 曲线
        ax = axes[0, 0]
        ax.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if self.history['val_loss'][0] > 0:
            ax.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Total Loss Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 步长 Loss 和重启 Loss 对比
        ax = axes[0, 1]
        ax.plot(epochs, self.history['train_step_loss'], 'g-', label='Step Loss', linewidth=2)
        ax.plot(epochs, self.history['train_restart_loss'], 'm-', label='Restart Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Component Loss Breakdown')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 学习率变化
        ax = axes[0, 2]
        ax.plot(epochs, self.history['learning_rate'], 'k-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # 4. 验证集分类指标
        if self.history['val_accuracy'][0] > 0:
            ax = axes[1, 0]
            ax.plot(epochs, self.history['val_accuracy'], 'b-o', label='Accuracy', linewidth=2)
            ax.plot(epochs, self.history['val_precision'], 'g-s', label='Precision', linewidth=2)
            ax.plot(epochs, self.history['val_recall'], 'r-^', label='Recall', linewidth=2)
            ax.plot(epochs, self.history['val_f1'], 'm-d', label='F1 Score', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score')
            ax.set_title('Validation Classification Metrics')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 5. Train vs Val Loss 对比
        if self.history['val_loss'][0] > 0:
            ax = axes[1, 1]
            ax.plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
            ax.plot(epochs, self.history['val_loss'], 'r-', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Train vs Validation Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 6. Loss 下降率 (对数坐标)
        ax = axes[1, 2]
        ax.semilogy(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        if self.history['val_loss'][0] > 0:
            ax.semilogy(epochs, self.history['val_loss'], 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (log scale)')
        ax.set_title('Loss Convergence (Log Scale)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = self.log_dir / 'learning_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"学习曲线已保存到：{save_path}")
        
        # 也记录到 TensorBoard
        self.writer.add_figure('learning_curves', fig, len(epochs))
        plt.close()
    
    def save_summary(self, final_metrics: dict = None):
        """
        保存训练摘要
        
        :param final_metrics: 最终指标字典
        """
        summary = {
            'history': self.history,
            'final_metrics': final_metrics,
            'total_epochs': len(self.history['train_loss'])
        }
        
        summary_path = self.log_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"训练摘要已保存到：{summary_path}")
    
    def close(self):
        """关闭 TensorBoard writer"""
        self.writer.close()
```

---

## 3. 训练脚本集成

### 3.1 完整训练脚本

```python
# scripts/train_controller.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
import argparse
import yaml

from src.nn.controller import NNController
from src.nn.loss import ControllerLoss
from src.train.data_generator import PseudospectrumDataset, DataGenerator
from src.train.logger import TrainingLogger
from src.core.manifold_ode import extract_features

def train(args):
    """主训练函数"""
    
    # === 1. 设置随机种子 ===
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # === 2. 初始化日志记录器 ===
    logger = TrainingLogger(
        log_dir='logs/',
        experiment_name=args.experiment_name
    )
    
    # 保存配置
    config = vars(args)
    with open(logger.config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # === 3. 加载或生成数据 ===
    if args.data_path:
        print(f"加载数据集：{args.data_path}")
        dataset = PseudospectrumDataset.load_dataset(
            args.data_path,
            args.norm_params_path
        )
    else:
        print("生成新数据集...")
        # 这里调用数据生成逻辑
        # 简化示例
        pass
    
    # === 4. 划分训练集/验证集 ===
    val_ratio = args.val_ratio
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"训练集：{len(train_dataset)} 样本")
    print(f"验证集：{len(val_dataset)} 样本")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)
    
    # === 5. 初始化模型 ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")
    
    model = NNController(
        input_dim=7,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout
    ).to(device)
    
    # === 6. 初始化优化器和损失 ===
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    loss_fn = ControllerLoss(
        lambda_step=args.lambda_step,
        lambda_restart=args.lambda_restart,
        alpha_restart=args.alpha_restart
    ).to(device)
    
    # === 7. 训练循环 ===
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        # --- 训练阶段 ---
        model.train()
        train_metrics = {'total_loss': 0, 'step_loss': 0, 'restart_loss': 0}
        num_train_batches = 0
        
        for batch in train_loader:
            features = batch['features'].to(device)
            ds_target = batch['ds_target'].to(device)
            y_restart = batch['y_restart'].to(device)
            
            # 前向传播
            ds_pred, p_restart = model(features)
            
            # 计算损失
            total_loss, step_loss, restart_loss = loss_fn(
                ds_pred.squeeze(), ds_target,
                p_restart.squeeze(), y_restart
            )
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 累积指标
            train_metrics['total_loss'] += total_loss.item()
            train_metrics['step_loss'] += step_loss.item()
            train_metrics['restart_loss'] += restart_loss.item()
            num_train_batches += 1
        
        # 平均训练损失
        for key in train_metrics:
            train_metrics[key] /= num_train_batches
        
        # --- 验证阶段 ---
        model.eval()
        val_metrics = {'total_loss': 0, 'step_loss': 0, 'restart_loss': 0}
        
        all_y_true = []
        all_y_pred = []
        all_ds_true = []
        all_ds_pred = []
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                ds_target = batch['ds_target'].to(device)
                y_restart = batch['y_restart'].to(device)
                
                ds_pred, p_restart = model(features)
                
                total_loss, step_loss, restart_loss = loss_fn(
                    ds_pred.squeeze(), ds_target,
                    p_restart.squeeze(), y_restart
                )
                
                val_metrics['total_loss'] += total_loss.item()
                val_metrics['step_loss'] += step_loss.item()
                val_metrics['restart_loss'] += restart_loss.item()
                
                # 收集预测结果
                all_y_true.extend(y_restart.cpu().numpy())
                all_y_pred.extend((p_restart.squeeze() > 0.5).cpu().numpy())
                all_ds_true.extend(batch['ds_raw'].numpy())
                all_ds_pred.extend(ds_pred.squeeze().cpu().numpy())
        
        # 平均验证损失
        num_val_batches = len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= num_val_batches
        
        # 计算分类指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        val_metrics['accuracy'] = accuracy_score(all_y_true, all_y_pred)
        val_metrics['precision'] = precision_score(all_y_true, all_y_pred, zero_division=0)
        val_metrics['recall'] = recall_score(all_y_true, all_y_pred, zero_division=0)
        val_metrics['f1'] = f1_score(all_y_true, all_y_pred, zero_division=0)
        
        # --- 记录日志 ---
        current_lr = optimizer.param_groups[0]['lr']
        logger.log_epoch(epoch, train_metrics, val_metrics, current_lr)
        
        # --- 学习率调度 ---
        scheduler.step(val_metrics['total_loss'])
        
        # --- 早停检查 ---
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            patience_counter = 0
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'args': args
            }, logger.log_dir / 'best_model.pth')
            print(f"✓ 保存最佳模型 (Val Loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"早停触发 (patience={args.early_stop_patience})")
                break
        
        # --- 定期可视化 ---
        if epoch % args.viz_interval == 0:
            # 混淆矩阵
            logger.log_confusion_matrix(
                np.array(all_y_true),
                np.array(all_y_pred),
                epoch,
                class_names=['No Restart', 'Restart']
            )
            
            # 步长预测散点图
            logger.log_prediction_scatter(
                np.array(all_ds_pred),
                np.array(all_ds_true),
                epoch
            )
            
            # 学习曲线
            logger.log_learning_curves()
    
    # === 8. 训练完成 ===
    logger.save_summary({
        'best_val_loss': best_val_loss,
        'final_train_loss': train_metrics['total_loss'],
        'final_val_loss': val_metrics['total_loss'],
        'final_val_f1': val_metrics['f1']
    })
    
    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args
    }, logger.log_dir / 'final_model.pth')
    
    logger.close()
    print(f"\n训练完成！日志保存在：{logger.log_dir}")
    
    return model, logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练神经网络控制器')
    
    # 数据参数
    parser.add_argument('--data-path', type=str, default=None,
                        help='数据集路径 (可选，不指定则生成新数据)')
    parser.add_argument('--norm-params-path', type=str, default=None,
                        help='归一化参数路径')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='验证集比例')
    
    # 模型参数
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[64, 64],
                        help='隐藏层维度列表')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout 比率')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='批大小')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='初始学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='L2 正则化')
    parser.add_argument('--early-stop-patience', type=int, default=15,
                        help='早停耐心值')
    
    # 损失参数
    parser.add_argument('--lambda-step', type=float, default=1.0,
                        help='步长损失权重')
    parser.add_argument('--lambda-restart', type=float, default=5.0,
                        help='重启损失权重')
    parser.add_argument('--alpha-restart', type=float, default=0.9,
                        help='重启正样本权重')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='实验名称')
    parser.add_argument('--viz-interval', type=int, default=5,
                        help='可视化间隔 (epoch)')
    
    args = parser.parse_args()
    train(args)
```

---

## 4. 可视化结果查看

### 4.1 TensorBoard 查看

```bash
# 启动 TensorBoard
tensorboard --logdir=logs/

# 在浏览器打开
# http://localhost:6006
```

**TensorBoard 面板内容**：
- **SCALARS**: Loss 曲线、学习率、分类指标
- **GRAPHS**: 模型计算图
- **HISTOGRAMS**: 权重和梯度分布
- **FIGURES**: 混淆矩阵、散点图、特征分布

### 4.2 生成的图像文件

训练完成后，在 `logs/{experiment_name}/` 目录下会生成：

```
logs/exp_20240101_120000/
├── best_model.pth           # 最佳模型检查点
├── final_model.pth          # 最终模型
├── config.json              # 训练配置
├── training_summary.json    # 完整训练历史
└── learning_curves.png      # 学习曲线汇总图
```

---

## 5. 训练后分析脚本

```python
# scripts/analyze_training.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import argparse

def load_training_results(log_dir: str):
    """加载训练结果"""
    log_path = Path(log_dir)
    
    # 加载训练历史
    with open(log_path / 'training_summary.json', 'r') as f:
        summary = json.load(f)
    
    # 加载最佳模型
    checkpoint = torch.load(log_path / 'best_model.pth')
    
    return summary, checkpoint

def plot_comparison(log_dirs: list, labels: list = None, 
                    save_path: str = 'training_comparison.png'):
    """
    比较多个实验的训练曲线
    
    :param log_dirs: 多个实验的日志目录列表
    :param labels: 实验标签列表
    :param save_path: 保存路径
    """
    if labels is None:
        labels = [Path(d).name for d in log_dirs]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(log_dirs)))
    
    for i, (log_dir, label) in enumerate(zip(log_dirs, labels)):
        summary, _ = load_training_results(log_dir)
        history = summary['history']
        epochs = range(1, len(history['train_loss']) + 1)
        
        # 1. 总 Loss 对比
        ax = axes[0, 0]
        ax.plot(epochs, history['train_loss'], '-', color=colors[i], 
                label=label, linewidth=2, alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Train Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 验证 Loss 对比
        ax = axes[0, 1]
        if history['val_loss'][0] > 0:
            ax.plot(epochs, history['val_loss'], '-', color=colors[i],
                    label=label, linewidth=2, alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. F1 Score 对比
        ax = axes[1, 0]
        if history['val_f1'][0] > 0:
            ax.plot(epochs, history['val_f1'], '-', color=colors[i],
                    label=label, linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val F1 Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 学习率对比
        ax = axes[1, 1]
        ax.plot(epochs, history['learning_rate'], '-', color=colors[i],
                label=label, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"对比图已保存到：{save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dirs', type=str, nargs='+',
                        help='实验日志目录列表')
    parser.add_argument('--labels', type=str, nargs='*', default=None,
                        help='实验标签')
    parser.add_argument('--output', type=str, default='training_comparison.png',
                        help='输出图像路径')
    
    args = parser.parse_args()
    plot_comparison(args.log_dirs, args.labels, args.output)
```

使用示例：

```bash
# 比较多个实验
python scripts/analyze_training.py \
    logs/exp_baseline logs/exp_lr1e-4 logs/exp_heavy_augment \
    --labels "Baseline" "LR=1e-4" "Heavy Augment" \
    --output comparison.png
```

---

## 6. 实时训练监控（可选）

### 6.1 Weights & Biases 集成

```python
# 在训练脚本开头添加
import wandb

wandb.init(
    project="pseudospectrum-controller",
    config=args,
    name=args.experiment_name
)

# 在每个 epoch 结束时
wandb.log({
    'train_loss': train_metrics['total_loss'],
    'val_loss': val_metrics['total_loss'],
    'val_f1': val_metrics['f1'],
    'epoch': epoch
})

wandb.finish()
```

启动：

```bash
wandb login
python scripts/train_controller.py --experiment-name my_exp
```

然后在 https://wandb.ai 查看实时面板。

---

## 7. 可视化检查清单

训练完成后，确认以下可视化已生成：

- [ ] **Loss 曲线**: 训练/验证 Loss 随 epoch 变化
- [ ] **组件 Loss**: 步长 Loss 和重启 Loss 分别的曲线
- [ ] **学习率曲线**: LR 调度可视化
- [ ] **混淆矩阵**: 重启决策的 TP/FP/TN/FN
- [ ] **步长散点图**: 预测 vs 真实步长对比
- [ ] **特征分布**: 7 维特征的直方图
- [ ] **分类指标**: Accuracy/Precision/Recall/F1 曲线
- [ ] **学习曲线汇总图**: `learning_curves.png`
