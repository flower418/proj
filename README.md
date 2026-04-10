# 神经增强子空间追踪算法

这个项目做的是：

- 给定矩阵 `A`
- 从复平面上的随机点或指定点出发
- 追踪经过该点的 `epsilon`-伪谱等高线

当前主线只有四个入口：

- `scripts/generate_large_dataset.py`
- `scripts/train_from_dataset.py`
- `scripts/benchmark_nn_vs_newton.py`
- `scripts/run_tracking.py`

## 快速开始

### 1. 安装环境

```bash
conda create -n pseudospectrum python=3.10 -y
conda activate pseudospectrum
pip install -r requirements.txt
```

### 2. 生成训练数据

当前生成策略固定为：

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

### 3. 检查数据

```bash
python src/data/dataset.py --data-dir data/my_data
```

### 4. 训练控制器

当前训练只读一个配置文件：`configs/default.yaml`。
不再有单独的 `training.yaml`。

```bash
python scripts/train_from_dataset.py \
    --data-dir data/my_data \
    --experiment-name my_model \
    --epochs 50 \
    --batch-size 256
```

训练输出：

- `models/my_model/best_model.pt`
- `models/my_model/training_history.json`
- `models/my_model/test_metrics.json`
- `logs/my_model/training_summary.png`

### 5. 跑 NN vs Newton benchmark

```bash
python scripts/benchmark_nn_vs_newton.py \
    --checkpoint models/my_model/best_model.pt \
    --matrix-size 20 \
    --seed 0 \
    --max-steps 16000 \
    --output-dir results/bench_demo
```

benchmark 默认串行执行，先跑 `NN + ODE`，再跑 `Newton PC`。

Benchmark 输出包括：

- `results/bench_demo/comparison_plot.png`
- `results/bench_demo/comparison_summary.json`
- `results/bench_demo/trajectories.npz`
- `results/bench_demo/random_matrix.npy`
- `results/bench_demo/logs/.../nn/summary.json`

其中 benchmark 只额外保留 NN 的 summary 文件，不再额外写 baseline summary 或 step 日志。

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
│   └── default.yaml
├── scripts/
│   ├── generate_large_dataset.py
│   ├── train_from_dataset.py
│   ├── benchmark_nn_vs_newton.py
│   └── run_tracking.py
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

训练标签来自高精度专家策略和 `DAgger` 增强。

## 测试

```bash
pytest tests -v
```

## 文档

- [SERVER_TUTORIAL.md](SERVER_TUTORIAL.md)
- [DATASET_EXPLANATION.md](DATASET_EXPLANATION.md)
