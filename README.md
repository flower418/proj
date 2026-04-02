# 神经增强子空间追踪算法

这个项目做的是：

- 给定矩阵 `A`
- 给定一个起点 `z0`
- 追踪经过该点的 `epsilon`-伪谱等高线

核心不是黑盒补轮廓，而是：

- 白盒 ODE 负责沿等高线推进
- 神经网络只负责预测步长 `ds` 和是否做一次精确 SVD 重启

如果你想从任意复平面点出发，最直接的方式是先计算

`epsilon = sigma_min(z0 I - A)`

然后追踪这一条经过 `z0` 的闭合等高线。项目里的 `scripts/demo_random_inference.py` 就是按这个逻辑工作的。

## 推荐工作流

当前最推荐、最完整的主流程是：

1. 用 `scripts/generate_large_dataset.py` 生成离线数据集
2. 用 `scripts/train_from_dataset.py` 训练控制器
3. 用 `scripts/evaluate.py` 或 `scripts/demo_random_inference.py` 做评估和推理

`scripts/generate_data.py` 和 `scripts/train_controller.py` 仍然保留，但它们更适合小规模 smoke test，不是主实验流程。

## 快速开始

### 1. 安装环境

```bash
conda create -n pseudospectrum python=3.10 -y
conda activate pseudospectrum
pip install -r requirements.txt
```

### 2. 生成训练数据

```bash
python scripts/generate_large_dataset.py \
    --target-samples 10000 \
    --matrix-sizes 30 50 80 \
    --output-dir data/my_data \
    --seed 0
```

生成结果默认包括：

- `data/my_data/dataset_full.npz`
- `data/my_data/dataset_full_splits.npz`
- `data/my_data/dataset_stats.json`

### 3. 训练控制器

```bash
python scripts/train_from_dataset.py \
    --data-dir data/my_data \
    --experiment-name my_model \
    --epochs 50 \
    --batch-size 256 \
    --device cuda
```

训练输出：

- `models/my_model/best_model.pt`
- `models/my_model/training_history.json`
- `models/my_model/test_metrics.json`
- `logs/my_model/training_summary.png`

### 4. 从随机点定义一条等高线并闭合追踪

```bash
python scripts/demo_random_inference.py \
    --checkpoint models/my_model/best_model.pt \
    --matrix-size 20 \
    --seed 0 \
    --output-dir results/random_demo
```

这个脚本会：

- 随机生成一个矩阵 `A`
- 随机抽取一个复平面点 `z_random`
- 计算 `epsilon = sigma_min(z_random I - A)`
- 从该点对应的等高线出发追踪完整闭合轮廓
- 保存图像和摘要 JSON

输出文件：

- `results/random_demo/random_matrix.npy`
- `results/random_demo/tracked_contour.png`
- `results/random_demo/tracking_summary.json`

### 5. 对你自己的矩阵做追踪

如果你已经有矩阵文件：

```bash
python scripts/run_tracking.py \
    --matrix-path path/to/matrix.npy \
    --checkpoint models/my_model/best_model.pt \
    --plot-out results/trajectory.png \
    --result-out results/trajectory.json \
    --epsilon 0.1
```

说明：

- 如果你显式给 `--epsilon`，脚本追踪的是 `sigma_min(zI-A)=epsilon` 这条等高线
- 如果你再给 `--z0-real` 和 `--z0-imag`，脚本会先把这个猜测点投影到真实等高线上
- 如果你不提供 `z0`，脚本会自动从极值特征值附近选一个起点

## 项目结构

```text
proj/
├── configs/
│   ├── default.yaml
│   └── training.yaml
├── scripts/
│   ├── generate_large_dataset.py
│   ├── train_from_dataset.py
│   ├── evaluate.py
│   ├── demo_random_inference.py
│   ├── run_tracking.py
│   ├── generate_data.py
│   └── train_controller.py
├── src/
│   ├── core/
│   │   ├── manifold_ode.py
│   │   ├── pseudoinverse.py
│   │   └── contour_tracker.py
│   ├── data/
│   │   └── dataset.py
│   ├── nn/
│   │   ├── controller.py
│   │   ├── features.py
│   │   └── loss.py
│   ├── train/
│   │   ├── expert_solver.py
│   │   ├── dagger_augmentation.py
│   │   ├── data_generator.py
│   │   ├── trainer.py
│   │   └── logger.py
│   ├── solvers/
│   │   └── rk4.py
│   └── utils/
│       ├── contour_init.py
│       ├── svd.py
│       ├── metrics.py
│       ├── visualization.py
│       └── config.py
└── tests/
```

## 核心算法

### 1. ODE 底座

在等高线 `sigma_min(zI-A)=epsilon` 上追踪状态 `(z, u, v)`，其中：

- `z` 是复平面位置
- `u, v` 是最小奇异值对应的左/右奇异向量

推进由 `src/core/manifold_ode.py` 实现。

### 2. 控制器输入

网络输入是 7 维标量特征：

- 奇异值偏差
- 向量模长漂移
- 残差范数
- 梯度模长
- 曲率代理量
- 伪逆求解器迭代次数

这些特征与矩阵维度无关。

### 3. 控制器输出

网络输出两个量：

- `ds`：下一步步长
- `y_restart`：是否做一次精确 SVD 重启

### 4. 专家数据

训练标签来自高精度专家策略：

- 专家推进器：`RK45`
- 专家重启：残差/漂移阈值触发
- 数据增强：`DAgger`

## 重要实现约定

- `demo_random_inference.py` 在 `sample_mode=point_sigma` 下，追踪的是“经过随机点的那条等高线”
- `run_tracking.py` 在固定 `epsilon` 下，若用户给的是一个猜测点，会先投影到真实等高线
- 默认 `tracker.max_steps=4000`，因为闭合检测对步长敏感，预算过小会让本应闭合的轨迹提前停止

## 默认配置摘要

`configs/default.yaml` 当前关键默认值：

```yaml
ode:
  epsilon: 0.1
  initial_step_size: 0.01
  min_step_size: 1.0e-6
  max_step_size: 0.1

solver:
  method: minres
  tol: 1.0e-8
  max_iter: 500

tracker:
  max_steps: 4000
  closure_tol: 1.0e-3
  restart_drift_threshold: 1.0e-4

controller:
  hidden_dims: [64, 64]
  dropout: 0.1
  norm_type: layernorm
  step_size_min: 1.0e-4
  step_size_max: 0.1
```

## 测试

```bash
pytest tests -v
```

## 文档

- [SERVER_TUTORIAL.md](SERVER_TUTORIAL.md)
- [DATASET_EXPLANATION.md](DATASET_EXPLANATION.md)

## 说明

- 一次运行只追踪一个连通等高线分量
- 起点决定你追踪的是哪一条分量
- 神经网络不会替代伪谱定义，只是在数值推进时做控制

## License

MIT
