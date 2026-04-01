# 完整使用指南 - 测试、数据生成、训练、推理

本文档提供从零开始搭建、测试、训练到推理的完整流程指导。

---

## 目录

1. [环境搭建](#1-环境搭建)
2. [项目结构](#2-项目结构)
3. [单元测试](#3-单元测试)
4. [数据生成](#4-数据生成)
5. [模型训练](#5-模型训练)
6. [模型评估](#6-模型评估)
7. [推理使用](#7-推理使用)
8. [故障排查](#8-故障排查)

---

## 1. 环境搭建

### 1.1 创建虚拟环境

```bash
# 使用 conda (推荐)
conda create -n pseudospectrum python=3.10
conda activate pseudospectrum

# 或使用 venv
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

### 1.2 安装依赖

```bash
# 创建 requirements.txt
cat > requirements.txt << EOF
# 核心科学计算
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0

# 深度学习
torch>=2.0.0
torchvision>=0.15.0

# 工具
pyyaml>=6.0
tensorboard>=2.12.0
scikit-learn>=1.2.0
tqdm>=4.65.0

# 开发测试
pytest>=7.3.0
pytest-cov>=4.0.0
black>=23.0.0
flake8>=6.0.0
EOF

# 安装
pip install -r requirements.txt
```

### 1.3 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import scipy; print(f'SciPy: {scipy.__version__}')"
```

---

## 2. 项目结构

按照以下方式组织项目：

```
pseudospectrum_tracker/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   ├── default.yaml
│   └── training.yaml
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── manifold_ode.py
│   │   ├── pseudoinverse.py
│   │   └── contour_tracker.py
│   ├── nn/
│   │   ├── __init__.py
│   │   ├── controller.py
│   │   ├── features.py
│   │   └── loss.py
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── rk4.py
│   │   └── krylov.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── svd.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── train/
│       ├── __init__.py
│       ├── data_generator.py
│       ├── trainer.py
│       └── logger.py
├── tests/
│   ├── test_manifold_ode.py
│   ├── test_controller.py
│   ├── test_features.py
│   └── test_integration.py
├── scripts/
│   ├── generate_data.py
│   ├── train_controller.py
│   ├── evaluate.py
│   ├── run_tracking.py
│   └── analyze_training.py
├── logs/              # 训练日志 (gitignore)
├── data/              # 生成的数据 (gitignore)
└── models/            # 保存的模型 (gitignore)
```

---

## 3. 单元测试

### 3.1 运行所有测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行并生成覆盖率报告
pytest tests/ -v --cov=src --cov-report=html

# 运行特定测试文件
pytest tests/test_manifold_ode.py -v
```

### 3.2 核心测试用例实现

#### 测试 ODE 系统 (`tests/test_manifold_ode.py`)

```python
import numpy as np
import pytest
from src.core.manifold_ode import ManifoldODE

class TestManifoldODE:
    """测试微分流形 ODE 系统"""
    
    @pytest.fixture
    def setup_small_matrix(self):
        """准备小规模测试矩阵"""
        np.random.seed(42)
        n = 50
        A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        epsilon = 0.1
        return A, epsilon, n
    
    def test_dz_ds_unit_norm(self, setup_small_matrix):
        """验证 dz/ds 的模长恒为 1"""
        A, epsilon, n = setup_small_matrix
        ode = ManifoldODE(A, epsilon)
        
        # 随机生成单位奇异向量
        np.random.seed(123)
        u = np.random.randn(n) + 1j * np.random.randn(n)
        u = u / np.linalg.norm(u)
        v = np.random.randn(n) + 1j * np.random.randn(n)
        v = v / np.linalg.norm(v)
        z = 1.0 + 0.5j
        
        dz_ds = ode.compute_dz_ds(z, u, v)
        
        # 验证模长为 1 (允许数值误差)
        assert np.abs(np.abs(dz_ds) - 1.0) < 1e-10, \
            f"|dz/ds| = {np.abs(dz_ds)}, 应该等于 1"
    
    def test_dv_orthogonal_to_v(self, setup_small_matrix):
        """验证 dv/ds ⟂ v"""
        A, epsilon, n = setup_small_matrix
        ode = ManifoldODE(A, epsilon)
        
        np.random.seed(123)
        u = np.random.randn(n) + 1j * np.random.randn(n)
        u = u / np.linalg.norm(u)
        v = np.random.randn(n) + 1j * np.random.randn(n)
        v = v / np.linalg.norm(v)
        z = 1.0 + 0.5j
        
        dz_ds = ode.compute_dz_ds(z, u, v)
        dv_ds = ode.compute_dv_ds(z, u, v, dz_ds)
        
        # 验证正交性 Re(v* · dv/ds) = 0
        orthogonality_error = np.abs(np.real(np.vdot(v, dv_ds)))
        assert orthogonality_error < 1e-10, \
            f"正交性误差 = {orthogonality_error}"
    
    def test_sigma_conservation(self, setup_small_matrix):
        """验证沿轨线 σ(z) ≈ epsilon"""
        A, epsilon, n = setup_small_matrix
        ode = ManifoldODE(A, epsilon)
        
        # 使用精确 SVD 初始化
        z = 1.0 + 0.5j
        M = z * np.eye(n) - A
        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        idx = np.argmin(S)
        u = U[:, idx]
        v = Vh[idx, :].conj().T
        
        # 沿轨线走 10 步
        ds = 0.01
        for _ in range(10):
            dz_ds, du_ds, dv_ds = ode.get_full_derivatives(z, u, v)
            z = z + ds * dz_ds
            u = u + ds * du_ds
            v = v + ds * dv_ds
            
            # 归一化
            u = u / np.linalg.norm(u)
            v = v / np.linalg.norm(v)
            
            # 检查奇异值
            M = z * np.eye(n) - A
            sigma_approx = np.abs(np.vdot(u, M @ v))
            assert np.abs(sigma_approx - epsilon) < 1e-3, \
                f"σ = {sigma_approx}, 应该接近 {epsilon}"
```

#### 测试神经网络 (`tests/test_controller.py`)

```python
import torch
import numpy as np
import pytest
from src.nn.controller import NNController

class TestNNController:
    """测试神经网络控制器"""
    
    @pytest.fixture
    def setup_controller(self):
        """准备控制器模型"""
        return NNController(
            input_dim=7,
            hidden_dims=[32, 32],
            dropout=0.1
        )
    
    def test_output_positive_step_size(self, setup_controller):
        """验证步长输出严格为正"""
        controller = setup_controller
        controller.eval()
        
        # 随机输入
        x = torch.randn(10, 7)
        
        with torch.no_grad():
            ds_pred, _ = controller(x)
        
        # 验证所有输出 > 0
        assert torch.all(ds_pred > 0), "步长输出必须为正"
    
    def test_restart_probability_range(self, setup_controller):
        """验证重启概率 ∈ (0, 1)"""
        controller = setup_controller
        controller.eval()
        
        x = torch.randn(10, 7)
        
        with torch.no_grad():
            _, p_restart = controller(x)
        
        assert torch.all(p_restart >= 0) and torch.all(p_restart <= 1), \
            "重启概率必须在 [0, 1] 范围内"
    
    def test_parameter_count(self, setup_controller):
        """验证模型参数量"""
        total_params = sum(p.numel() for p in setup_controller.parameters())
        trainable_params = sum(p.numel() for p in setup_controller.parameters() 
                               if p.requires_grad)
        
        assert total_params == trainable_params, "所有参数应该都可训练"
        print(f"总参数量：{total_params:,}")
```

#### 测试特征提取 (`tests/test_features.py`)

```python
import numpy as np
import pytest
from src.nn.features import extract_features

class TestFeatureExtraction:
    """测试状态特征提取"""
    
    @pytest.fixture
    def setup_state(self):
        """准备测试状态"""
        np.random.seed(42)
        n = 50
        A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        z = 1.0 + 0.5j
        
        M = z * np.eye(n) - A
        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        idx = np.argmin(S)
        u = U[:, idx]
        v = Vh[idx, :].conj().T
        epsilon = 0.1
        
        return A, z, u, v, epsilon
    
    def test_feature_dimension(self, setup_state):
        """验证特征维度为 7"""
        A, z, u, v, epsilon = setup_state
        
        features = extract_features(z, u, v, A, epsilon)
        
        assert features.shape == (7,), f"特征维度应该是 7, 得到 {features.shape}"
    
    def test_feature_values_finite(self, setup_state):
        """验证所有特征值有限"""
        A, z, u, v, epsilon = setup_state
        
        features = extract_features(z, u, v, A, epsilon)
        
        assert np.all(np.isfinite(features)), "特征值必须有限"
    
    def test_f5_gradient_norm_positive(self, setup_state):
        """验证 f5 (梯度模长) 非负"""
        A, z, u, v, epsilon = setup_state
        
        features = extract_features(z, u, v, A, epsilon)
        f5 = features[4]
        
        assert f5 >= 0, "f5 (梯度模长) 必须非负"
```

#### 集成测试 (`tests/test_integration.py`)

```python
import numpy as np
import pytest
from src.core.contour_tracker import ContourTracker
from src.core.manifold_ode import ManifoldODE

class TestIntegration:
    """集成测试 - 完整追踪流程"""
    
    def test_closed_contour_tracking(self):
        """测试闭合等高线追踪"""
        np.random.seed(42)
        n = 50
        A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        epsilon = 0.1
        
        # 找到等高线上的起点
        z0 = 1.0 + 0.5j  # 实际应用中需要通过搜索确定
        
        ode = ManifoldODE(A, epsilon)
        tracker = ContourTracker(A, epsilon, ode)
        
        # 运行追踪
        result = tracker.track(z0, max_steps=200)
        
        # 验证轨迹闭合
        z_start = result['trajectory'][0]
        z_end = result['trajectory'][-1]
        closure_error = np.abs(z_start - z_end)
        
        print(f"闭合误差：{closure_error:.6f}")
        assert closure_error < 0.1, f"轨迹应该闭合，闭合误差 = {closure_error}"
        
        # 验证至少走了一圈
        assert len(result['trajectory']) > 10, "轨迹应该有一定长度"
```

---

## 4. 数据生成

### 4.1 快速生成（单条轨迹）

```bash
python scripts/generate_data.py \
    --matrix-size 100 \
    --epsilon 0.1 \
    --max-steps 200 \
    --perturbations 3 \
    --output data/single_trajectory
```

### 4.2 批量生成（多条轨迹）

```bash
python scripts/generate_data.py \
    --matrix-size 100 \
    --epsilon 0.1 \
    --num-starting-points 8 \
    --max-steps 300 \
    --perturbations 5 \
    --output data/multiple_trajectories
```

### 4.3 数据生成脚本详解

```python
# scripts/generate_data.py

import numpy as np
import argparse
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.manifold_ode import extract_features
from src.train.data_generator import DataGenerator

def find_contour_points(A, epsilon, num_points=8):
    """
    在伪谱等高线上找到多个起点
    """
    from scipy.optimize import brentq
    
    n = A.shape[0]
    points = []
    
    for k in range(num_points):
        angle = 2 * np.pi * k / num_points
        
        def objective(r):
            z = r * np.exp(1j * angle)
            M = z * np.eye(n) - A
            # 近似最小奇异值
            sigma_min = np.linalg.norm(np.linalg.solve(M, np.random.randn(n)))
            return sigma_min - epsilon
        
        try:
            r_sol = brentq(objective, 0.1, 5.0)
            points.append(r_sol * np.exp(1j * angle))
            print(f"  起点 {k+1}/{num_points}: z = {r_sol * np.exp(1j * angle):.4f}")
        except ValueError:
            print(f"  起点 {k+1}/{num_points}: 未找到解，跳过")
    
    return points

def main():
    parser = argparse.ArgumentParser(description='生成训练数据')
    
    # 矩阵参数
    parser.add_argument('--matrix-size', type=int, default=100,
                        help='矩阵维度 n')
    parser.add_argument('--matrix-type', type=str, default='random_complex',
                        choices=['random_complex', 'random_hermitian', 'random_real'],
                        help='矩阵类型')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='伪谱水平')
    
    # 数据生成参数
    parser.add_argument('--num-starting-points', type=int, default=1,
                        help='起点数量 (1=单条轨迹)')
    parser.add_argument('--max-steps', type=int, default=200,
                        help='每条轨迹最大步数')
    parser.add_argument('--perturbations', type=int, default=3,
                        help='每个点的扰动样本数')
    
    # 输出
    parser.add_argument('--output', type=str, default='data/generated',
                        help='输出目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 生成矩阵
    print(f"生成 {args.matrix_size}x{args.matrix_size} 矩阵...")
    if args.matrix_type == 'random_complex':
        A = np.random.randn(args.matrix_size, args.matrix_size) + \
            1j * np.random.randn(args.matrix_size, args.matrix_size)
    elif args.matrix_type == 'random_hermitian':
        R = np.random.randn(args.matrix_size, args.matrix_size) + \
            1j * np.random.randn(args.matrix_size, args.matrix_size)
        A = (R + R.conj().T) / 2
    else:  # random_real
        A = np.random.randn(args.matrix_size, args.matrix_size)
    
    print(f"矩阵条件数估计：{np.linalg.cond(A):.2e}")
    
    # 初始化生成器
    generator = DataGenerator(
        A=A,
        epsilon=args.epsilon,
        feature_extractor=extract_features,
        save_dir=args.output
    )
    
    # 生成数据
    if args.num_starting_points == 1:
        # 单条轨迹
        print("\n生成单条轨迹...")
        z0 = 1.0 + 0.5j  # 简单示例，实际应该搜索
        dataset = generator.generate_from_matrix(
            z0=z0,
            max_steps=args.max_steps,
            num_perturbations=args.perturbations,
            save=True
        )
    else:
        # 多条轨迹
        print(f"\n生成 {args.num_starting_points} 条轨迹...")
        z0_list = find_contour_points(A, args.epsilon, args.num_starting_points)
        
        if len(z0_list) == 0:
            print("错误：未找到任何起点")
            return
        
        dataset = generator.generate_multiple_trajectories(
            z0_list=z0_list,
            max_steps=args.max_steps,
            num_perturbations=args.perturbations,
            save=True
        )
    
    # 打印统计信息
    print(f"\n{'='*50}")
    print("数据生成完成!")
    print(f"{'='*50}")
    print(f"总样本数：{len(dataset):,}")
    print(f"特征维度：{dataset[0]['features'].shape}")
    
    # 类别分布
    restart_count = sum(1 for d in dataset.data if d['y_restart'] == 1)
    print(f"重启样本比例：{restart_count}/{len(dataset.data)} = {restart_count/len(dataset.data):.2%}")

if __name__ == '__main__':
    main()
```

### 4.4 数据分析

```bash
python scripts/analyze_data.py data/generated/
```

---

## 5. 模型训练

### 5.1 开始训练

```bash
python scripts/train_controller.py \
    --data-path data/multiple_trajectories/dataset_*.pt \
    --norm-params-path data/multiple_trajectories/norm_params_*.json \
    --epochs 100 \
    --batch-size 128 \
    --learning-rate 1e-3 \
    --experiment-name baseline_experiment
```

### 5.2 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 100 | 训练轮数 |
| `--batch-size` | 128 | 批大小 |
| `--learning-rate` | 1e-3 | 初始学习率 |
| `--hidden-dims` | [64, 64] | 隐藏层维度 |
| `--dropout` | 0.1 | Dropout 比率 |
| `--lambda-step` | 1.0 | 步长损失权重 |
| `--lambda-restart` | 5.0 | 重启损失权重 |
| `--alpha-restart` | 0.9 | 重启正样本权重 |
| `--early-stop-patience` | 15 | 早停耐心值 |
| `--viz-interval` | 5 | 可视化间隔 (epoch) |

### 5.3 超参数调优建议

```bash
# 实验 1: 学习率对比
for lr in 1e-3 5e-4 1e-4; do
    python scripts/train_controller.py --learning-rate $lr \
        --experiment-name "lr_${lr}"
done

# 实验 2: 网络深度对比
for dims in "32" "64,64" "128,128,128"; do
    python scripts/train_controller.py --hidden-dims $dims \
        --experiment-name "depth_${dims//,/_}"
done

# 实验 3: 重启损失权重对比
for weight in 1.0 5.0 10.0; do
    python scripts/train_controller.py --lambda-restart $weight \
        --experiment-name "restart_weight_${weight}"
done
```

### 5.4 监控训练

```bash
# 启动 TensorBoard
tensorboard --logdir=logs/

# 浏览器打开 http://localhost:6006
```

---

## 6. 模型评估

### 6.1 在测试集上评估

```bash
python scripts/evaluate.py \
    --model-path logs/baseline_experiment/best_model.pth \
    --data-path data/multiple_trajectories/dataset_*.pt \
    --norm-params-path data/multiple_trajectories/norm_params_*.json
```

### 6.2 评估脚本

```python
# scripts/evaluate.py

import torch
import numpy as np
import argparse
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from src.nn.controller import NNController
from src.train.data_generator import PseudospectrumDataset

def evaluate(args):
    # 加载数据
    dataset = PseudospectrumDataset.load_dataset(args.data_path, args.norm_params_path)
    
    # 加载模型
    checkpoint = torch.load(args.model_path)
    model = NNController()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 推理
    all_y_true = []
    all_y_pred = []
    all_ds_true = []
    all_ds_pred = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            features = item['features'].unsqueeze(0)
            
            ds_pred, p_restart = model(features)
            
            all_y_true.append(item['y_restart'].item())
            all_y_pred.append((p_restart.squeeze() > 0.5).item())
            all_ds_true.append(item['ds_raw'].item())
            all_ds_pred.append(ds_pred.squeeze().item())
    
    # 计算指标
    print("\n" + "="*50)
    print("评估结果")
    print("="*50)
    print(f"重启决策准确率：{accuracy_score(all_y_true, all_y_pred):.4f}")
    print(f"重启决策精确率：{precision_score(all_y_true, all_y_pred, zero_division=0):.4f}")
    print(f"重启决策召回率：{recall_score(all_y_true, all_y_pred, zero_division=0):.4f}")
    print(f"重启决策 F1 分数：{f1_score(all_y_true, all_y_pred, zero_division=0):.4f}")
    
    # 步长预测误差
    ds_true = np.array(all_ds_true)
    ds_pred = np.array(all_ds_pred)
    mask = ds_true > 0
    mape = np.mean(np.abs((ds_true[mask] - ds_pred[mask]) / ds_true[mask])) * 100
    print(f"\n步长预测 MAPE: {mape:.2f}%")
    
    # 绘制混淆矩阵
    cm = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=20)
    
    plt.xticks([0, 1], ['No Restart', 'Restart'])
    plt.yticks([0, 1], ['No Restart', 'Restart'])
    plt.tight_layout()
    
    save_path = Path(args.model_path).parent / 'confusion_matrix.png'
    plt.savefig(save_path, dpi=150)
    print(f"\n混淆矩阵已保存到：{save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--norm-params-path', type=str, required=True)
    args = parser.parse_args()
    evaluate(args)
```

---

## 7. 推理使用

### 7.1 运行追踪

```bash
python scripts/run_tracking.py \
    --matrix data/my_matrix.npy \
    --epsilon 0.1 \
    --z0 1.0+0.5j \
    --model-path logs/baseline_experiment/best_model.pth \
    --output results/trajectory.png
```

### 7.2 推理脚本

```python
# scripts/run_tracking.py

import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

from src.core.contour_tracker import ContourTracker
from src.core.manifold_ode import ManifoldODE
from src.nn.controller import NNController
from src.nn.features import extract_features

def run_tracking(args):
    # 加载矩阵
    A = np.load(args.matrix)
    print(f"矩阵形状：{A.shape}")
    
    # 解析起点
    z0 = complex(args.z0.replace('j', 'J'))
    
    # 加载模型
    checkpoint = torch.load(args.model_path)
    controller = NNController()
    controller.load_state_dict(checkpoint['model_state_dict'])
    controller.eval()
    
    # 初始化追踪器
    ode = ManifoldODE(A, args.epsilon)
    tracker = ContourTracker(
        A, args.epsilon, ode,
        controller=controller
    )
    
    # 运行追踪
    print(f"从 z0 = {z0} 开始追踪...")
    result = tracker.track(z0, max_steps=args.max_steps)
    
    print(f"追踪完成！轨迹点数：{len(result['trajectory'])}")
    print(f"SVD 重启次数：{len(result['restart_indices'])}")
    
    # 可视化
    trajectory = np.array(result['trajectory'])
    
    plt.figure(figsize=(10, 8))
    plt.plot(trajectory.real, trajectory.imag, 'b-', linewidth=1, label='Contour')
    plt.plot(trajectory[0].real, trajectory[0].imag, 'go', markersize=10, label='Start')
    plt.plot(trajectory[-1].real, trajectory[-1].imag, 'rs', markersize=10, label='End')
    
    # 标记重启点
    if result['restart_indices']:
        restart_points = [trajectory[i] for i in result['restart_indices']]
        restart_real = [p.real for p in restart_points]
        restart_imag = [p.imag for p in restart_points]
        plt.scatter(restart_real, restart_imag, c='orange', s=50, 
                   marker='x', label='SVD Restart')
    
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.title(f'Pseudospectrum Contour (ε={args.epsilon})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"轨迹图已保存到：{output_path}")
    
    # 保存轨迹数据
    np.save(output_path.with_suffix('.npy'), trajectory)
    print(f"轨迹数据已保存到：{output_path.with_suffix('.npy')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', type=str, required=True,
                        help='矩阵文件路径 (.npy)')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='伪谱水平')
    parser.add_argument('--z0', type=str, required=True,
                        help='起点 (复数格式，如 "1.0+0.5j")')
    parser.add_argument('--model-path', type=str, required=True,
                        help='训练好的模型路径')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='最大追踪步数')
    parser.add_argument('--output', type=str, default='results/trajectory.png',
                        help='输出图像路径')
    
    args = parser.parse_args()
    run_tracking(args)
```

---

## 8. 故障排查

### 8.1 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 测试失败 `test_dz_ds_unit_norm` | 数值精度问题 | 检查容差设置，确认奇异向量归一化 |
| 训练 Loss 不下降 | 学习率过大/过小 | 尝试 1e-4, 5e-4, 1e-3 对比 |
| 重启样本过少 (<1%) | drift_threshold 太严格 | 从 1e-4 放宽到 1e-3 |
| CUDA Out of Memory | 批大小过大 | 减小 batch_size 到 64 或 32 |
| 轨迹不闭合 | 起点不在等高线上 | 用 Brent 搜索精确起点 |
| 推理时轨迹发散 | 模型泛化差 | 增加 DAgger 扰动强度 |

### 8.2 调试技巧

```bash
# 单步调试
python -m pdb scripts/train_controller.py --epochs 1

# 检查梯度
# 在训练循环中添加:
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm():.6f}")
```

---

## 9. 完整流程检查清单

### 第一阶段：环境搭建
- [ ] 创建虚拟环境
- [ ] 安装依赖
- [ ] 验证 PyTorch/CUDA

### 第二阶段：代码实现
- [ ] 实现 `ManifoldODE` (Eq. 13)
- [ ] 实现 `PseudoinverseSolver`
- [ ] 实现 `NNController`
- [ ] 实现 `extract_features`
- [ ] 实现 `ControllerLoss`

### 第三阶段：测试
- [ ] 运行 `pytest tests/ -v`
- [ ] 所有单元测试通过
- [ ] 集成测试通过

### 第四阶段：数据生成
- [ ] 生成单条轨迹测试
- [ ] 生成 8+ 条轨迹大数据集
- [ ] 验证重启样本比例 1-10%
- [ ] 可视化分析数据分布

### 第五阶段：训练
- [ ] 启动训练
- [ ] TensorBoard 监控
- [ ] Loss 曲线正常下降
- [ ] 保存最佳模型

### 第六阶段：评估
- [ ] 在测试集评估
- [ ] F1 分数 > 0.7
- [ ] 步长 MAPE < 50%

### 第七阶段：推理
- [ ] 加载训练模型
- [ ] 运行追踪
- [ ] 可视化轨迹
- [ ] 与基线方法对比

---

祝你训练顺利！如有问题，查看 `logs/` 中的详细记录。
