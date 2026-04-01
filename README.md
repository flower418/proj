# 神经增强子空间追踪算法

基于深度学习的伪谱等高线追踪算法，通过神经网络预测最优步长和 SVD 重启时机，显著加速大规模矩阵的伪谱计算。

## 快速开始

### 1. 安装

```bash
conda create -n pseudospectrum python=3.10 -y
conda activate pseudospectrum
pip install -r requirements.txt
```

### 2. 生成数据

```bash
python scripts/generate_large_dataset.py \
    --target-samples 10000 \
    --matrix-sizes 30 50 80 \
    --output-dir data/my_data \
    --seed 0
```

### 3. 训练

```bash
python scripts/train_from_dataset.py \
    --data-dir data/my_data \
    --experiment-name my_model \
    --epochs 50 \
    --device cuda
```

### 4. 推理

```bash
python scripts/run_tracking.py \
    --matrix-path path/to/matrix.npy \
    --checkpoint models/my_model/best_model.pt \
    --plot-out results/trajectory.png \
    --epsilon 0.1
```

详细教程请查看：**[SERVER_TUTORIAL.md](SERVER_TUTORIAL.md)**

---

## 项目结构

```
proj/
├── scripts/
│   ├── generate_large_dataset.py  # 数据生成
│   ├── train_from_dataset.py      # 训练模型
│   ├── evaluate.py                # 评估模型
│   └── run_tracking.py            # 推理可视化
├── src/
│   ├── core/                      # 核心算法
│   │   ├── manifold_ode.py        # ODE 系统 (Eq. 13)
│   │   ├── pseudoinverse.py       # 伪逆求解器
│   │   └── contour_tracker.py     # 等高线追踪器
│   ├── nn/                        # 神经网络
│   │   ├── controller.py          # 控制器模型
│   │   ├── features.py            # 特征提取
│   │   └── loss.py                # 损失函数
│   ├── train/                     # 训练相关
│   │   ├── expert_solver.py       # 专家求解器 (RK45)
│   │   ├── dagger_augmentation.py # DAgger 增强
│   │   ├── trainer.py             # 训练循环
│   │   └── logger.py              # 本地训练日志与总图
│   ├── solvers/                   # 数值求解器
│   │   └── rk4.py                 # RK4 积分器
│   ├── utils/                     # 工具函数
│   │   ├── svd.py                 # SVD 工具
│   │   ├── metrics.py             # 评估指标
│   │   ├── visualization.py       # 可视化
│   │   └── config.py              # 配置加载
│   └── data/                      # 数据处理
│       └── dataset.py             # 数据集加载
├── configs/                       # 配置文件
│   ├── default.yaml               # 默认配置
│   └── training.yaml              # 训练配置
├── tests/                         # 单元测试
├── data/                          # 生成的数据
├── models/                        # 训练好的模型
├── logs/                          # 本地训练总图与历史
└── results/                       # 推理结果图
```

---

## 核心算法

### 微分流形 ODE 系统

追踪伪谱等高线的 ODE 系统（Eq. 13）：

```
dz/ds = i * (v*u) / |v*u|
dv/ds = -(M*M - ε²I)⁺ · (dz/ds · M*v + ε · d(z̄)/ds · u)
du/ds = -(MM* - ε²I)⁺ · (d(z̄)/ds · Mu + ε · dz/ds · v)
```

其中 `M = zI - A`，`(·)⁺` 表示 Moore-Penrose 伪逆。

### 神经网络控制器

输入：7 维特征（与矩阵维度无关）
- f1: 奇异值偏差
- f2, f3: 奇异向量模长漂移
- f4: 残差范数
- f5: 复梯度模长
- f6: 曲率
- f7: 伪逆求解迭代次数

输出：
- `ds`: 最优步长（回归）
- `y_restart`: 是否需要 SVD 重启（分类）

### 训练数据标注

使用**自适应 RK45 积分器**作为专家：
1. 初始标注：稠密 SVD (`np.linalg.svd`)
2. 轨迹生成：RK45 自适应选择步长
3. 重启决策：残差阈值检查
4. 数据增强：DAgger 扰动

---

## 配置说明

### `configs/default.yaml`

```yaml
ode:
  epsilon: 0.1                  # 伪谱水平
  initial_step_size: 0.01       # 初始步长
  min_step_size: 1.0e-6         # 最小步长
  max_step_size: 0.1            # 最大步长

solver:
  method: lgmres                # 伪逆求解方法
  tol: 1.0e-8                   # 容差
  max_iter: 500                 # 最大迭代

tracker:
  max_steps: 1000               # 最大追踪步数
  closure_tol: 1.0e-3           # 闭合检测容差
  restart_drift_threshold: 1.0e-4  # 重启阈值

controller:
  hidden_dims: [64, 64]         # 隐藏层维度
  dropout: 0.1
  norm_type: layernorm          # LayerNorm / BatchNorm
  step_size_min: 1.0e-4         # 最小输出步长
  step_size_max: 0.1            # 最大输出步长

training:
  batch_size: 128
  learning_rate: 1.0e-3
  epochs: 100
  lambda_step: 1.0              # 步长损失权重
  lambda_restart: 5.0           # 重启损失权重
  alpha_restart: 0.9            # 重启正样本权重
  focal_gamma: 2.0              # Focal Loss 参数
  noise_std: 0.01               # DAgger 噪声
```

---

## 测试

```bash
pytest tests/ -v
```

---

## 文档

- **[SERVER_TUTORIAL.md](SERVER_TUTORIAL.md)** - 服务器操作保姆级教程
- **[DATASET_EXPLANATION.md](DATASET_EXPLANATION.md)** - 数据集详细说明

---

## 引用

算法解决的是：
- 给定矩阵 `A`
- 给定伪谱水平 `epsilon`
- 给定边界上的起点 `z0`

然后沿着该矩阵的 `epsilon`-伪谱等高线追踪并闭合出完整轮廓。

它不是“仅凭几个点直接补全轮廓”的黑盒插值器。控制器学的是追踪过程中的步长与重启策略，不替代矩阵本身的伪谱定义。

补充说明：
- 一次运行只追踪一条连通等高线分量，不会一次性把所有分量全画完。
- 追踪哪一条分量，由起点 `z0` 决定。
- 如果你不显式提供 `z0`，`run_tracking.py` 会自动选择一个起点：默认取最右侧特征值对应分量的边界点。
- `z0-real` 和 `z0-imag` 组成复数 `z0 = z0_real + i z0_imag`，它表示复平面上的一个初始猜测点，不要求你正好落在等高线上；脚本会先把它投影到 `sigma_min(zI-A)=epsilon` 的真实边界上。

---

## License

MIT
