# 神经增强子空间追踪算法

这个项目做的是：

- 给定矩阵 `A`
- 从复平面上的随机点或指定点出发
- 追踪经过该点的 `epsilon`-伪谱等高线

核心不是黑盒补轮廓，而是：

- 白盒 ODE 负责沿等高线推进
- 神经网络只负责预测步长 `ds` 和是否做一次精确 SVD 重启

## 推荐工作流

当前建议的主流程只有这一条：

1. 用 `scripts/generate_large_dataset.py` 生成离线数据集
2. 用 `scripts/train_from_dataset.py` 训练控制器
3. 用 `scripts/evaluate.py`、`scripts/demo_random_inference.py`、`scripts/run_newton_baseline.py`、`scripts/benchmark_nn_vs_newton.py` 做评估和对比

`scripts/generate_data.py` 和 `scripts/train_controller.py` 只适合小规模 smoke test。

## 快速开始

### 1. 安装环境

```bash
conda create -n proj python=3.12 -y
conda activate proj
pip install -r requirements.txt
```

### 2. 生成训练数据

当前 `generate_large_dataset.py` 的策略固定为：

- 矩阵类型从内置支持集合中随机抽取
- 从谱中心附近的复高斯分布随机取点
- 用 `epsilon = sigma_min(zI - A)` 定义训练任务

示例：

```bash
python -u scripts/generate_large_dataset.py \
    --target-samples 10000 \
    --matrix-sizes 30 50 80 \
    --trajectories-per-type 5 \
    --max-steps 200 \
    --dagger-factor 2 \
    --output-dir data/my_data \
    --save-every 5000 \
    --seed 0
```

支持的默认矩阵类型目前包括：

- `random_complex`
- `random_hermitian`
- `random_real`
- `ill_conditioned`
- `random_normal`
- `banded_nonnormal`
- `low_rank_plus_noise`
- `jordan_perturbed`
- `block_structured`

生成结果默认包括：

- `data/my_data/dataset_full.npz`
- `data/my_data/dataset_full_splits.npz`
- `data/my_data/dataset_stats.json`
- `data/my_data/logs/...`

说明：

- `target_samples` 是下限，不是严格上限；脚本按整条轨迹追加样本，最终样本数可能略高于目标值
- train / val / test 划分优先按 `trajectory_id` 分组，避免同一轨迹同时进入不同 split

### 3. 检查数据

```bash
python src/data/dataset.py --data-dir data/my_data
```

当前数据集的控制器输入维度是 `14`。

### 4. 训练控制器

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

### 5. 从随机点做推理

```bash
python scripts/demo_random_inference.py \
    --checkpoint models/my_model/best_model.pt \
    --matrix-size 20 \
    --seed 0 \
    --output-dir results/random_demo
```

### 6. 对自己的矩阵做追踪

```bash
python scripts/run_tracking.py \
    --matrix-path path/to/matrix.npy \
    --checkpoint models/my_model/best_model.pt \
    --plot-out results/trajectory.png \
    --result-out results/trajectory.json \
    --epsilon 0.1
```

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
│   ├── benchmark_nn_vs_newton.py
│   ├── run_newton_baseline.py
│   ├── generate_data.py
│   └── train_controller.py
├── src/
└── tests/
```

## 核心算法

### 1. ODE 底座

在等高线 `sigma_min(zI-A)=epsilon` 上追踪状态 `(z, u, v)`，其中：

- `z` 是复平面位置
- `u, v` 是最小奇异值对应的左/右奇异向量

推进由 `src/core/manifold_ode.py` 实现。

### 2. 控制器输入

当前特征由 10 个基础特征和 4 个上下文特征拼接而成，总维度为 `14`。

### 3. 控制器输出

网络输出两个量：

- `ds`：下一步步长
- `y_restart`：是否做一次精确 SVD 重启

### 4. 专家数据

训练标签来自高精度专家策略：

- 专家推进器：高精度 ODE / 纠偏推进
- 专家重启：残差/漂移阈值触发
- 数据增强：`DAgger`
