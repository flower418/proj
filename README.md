# 神经增强子空间追踪算法

这个项目现在采用的是**纯步长控制**版本：

- 白盒几何 / ODE 负责给出沿等高线的推进方向
- 神经网络只预测下一步步长 `ds`
- 不再保留 restart 分支，也不再学习 restart 标签

目标任务是：

- 给定矩阵 `A`
- 从复平面上的一个起点 `z0` 出发
- 追踪经过该点的 `epsilon`-伪谱等高线 `sigma_min(zI-A)=epsilon`

---

## 当前推荐工作流

主流程只有三步：

1. `scripts/generate_large_dataset.py` 生成离线数据集
2. `scripts/train_from_dataset.py` 训练步长控制器
3. `scripts/run_tracking.py` / `scripts/benchmark_nn_vs_newton.py` 做推理和对比

---

## 快速开始

### 1. 安装环境

```bash
conda create -n proj python=3.12 -y
conda activate proj
pip install -r requirements.txt
```

### 2. 生成训练数据

```bash
python -u scripts/generate_large_dataset.py     --target-samples 10000     --matrix-sizes 30 50 80     --trajectories-per-type 5     --max-steps 200     --dagger-factor 2     --output-dir data/my_data     --save-every 5000     --seed 0
```

输出通常包括：

- `data/my_data/dataset_full.npz`
- `data/my_data/dataset_full_splits.npz`
- `data/my_data/dataset_stats.json`
- `data/my_data/logs/...`

说明：

- `target_samples` 是下限，不是严格上限
- 划分优先按 `trajectory_id` 分组，避免同一轨迹同时进入不同 split

### 3. 检查数据集

```bash
python src/data/dataset.py --data-dir data/my_data
```

当前控制器输入维度是 `8`。

### 4. 训练控制器

```bash
python scripts/train_from_dataset.py     --data-dir data/my_data     --experiment-name my_model     --epochs 50     --batch-size 256     --device cuda
```

训练输出：

- `models/my_model/best_model.pt`
- `models/my_model/training_history.json`
- `models/my_model/test_metrics.json`
- `logs/my_model/training_summary.png`

### 5. 对自己的矩阵做追踪

```bash
python scripts/run_tracking.py     --matrix-path path/to/matrix.npy     --checkpoint models/my_model/best_model.pt     --plot-out results/trajectory.png     --result-out results/trajectory.json     --epsilon 0.1
```

### 6. 与传统 Newton predictor-corrector 做对比

```bash
python scripts/benchmark_nn_vs_newton.py     --checkpoint models/my_model/best_model.pt     --output-dir results/nn_vs_newton
```

---

## 当前项目结构

```text
proj/
├── configs/
│   └── default.yaml
├── scripts/
│   ├── generate_large_dataset.py
│   ├── train_from_dataset.py
│   ├── run_tracking.py
│   └── benchmark_nn_vs_newton.py
├── src/
└── tests/
```

---

## 当前算法结构

### 1. ODE 底座

追踪状态是 `(z, u, v)`：

- `z`：复平面位置
- `u, v`：最小奇异值对应的左右奇异向量

几何方向由 `src/core/manifold_ode.py` 给出。

### 2. 控制器输入

当前输入总共 `8` 维：

- 6 个局部几何特征
- 2 个控制上下文特征

详细见 `NN_8维输入梳理.md`。

### 3. 控制器输出

网络只输出一个量：

- `ds`：下一步步长

### 4. 专家数据

训练标签来自高精度专家策略：

- 专家推进：full triplet RK4 + 投影 / 回退
- 标签：只保留 `ds_expert`
- 增强：`DAgger`

---

## 为什么这个版本仍然有效

一个容易产生误解的点是：

> 传统算法慢，是因为要频繁做 SVD；而“算梯度/切向”本身很便宜。你的方法如果还是会做 SVD，为什么还能更快？

关键不在于“把单次 SVD 变快”，而在于**减少高频、全量、严格的 SVD / 校正调用次数**。

当前实现真正节省时间的地方有三层：

1. **方向是白盒直接给的**
   - 切向方向由当前 `(u,v)` 直接决定
   - 这部分本来就便宜，不需要学习

2. **网络学习的是步长控制**
   - 在平滑、可信的区域敢走更大步
   - 在曲率大、残差大、刚投影过的区域主动缩步
   - 因此同一条轮廓通常可以用更少的总步数走完

3. **fast tangent tracker 会延迟或减少精确刷新**
   - 很多步只推进 `z`
   - `u,v` 尽量复用当前信息做近似判定
   - 只有必要时才做 projection 或 exact triplet refresh

所以它的收益来源不是：

- “神经网络替代了 SVD”

而是：

- “神经网络减少了你不得不做昂贵校正的频率，并减少了总步数”

也就是说，收益主要来自 **更好的控制策略**，不是来自更便宜的单步线性代数核。
