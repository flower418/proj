# 神经增强子空间追踪算法

这个项目当前采用的是**纯步长控制**版本：

- 白盒几何负责给出沿等高线的推进方向
- 神经网络只预测下一步步长 `ds`
- 不再保留 restart 分支
- 推理主链已收缩为单次接受结构

目标任务是追踪矩阵 `A` 的 `epsilon`-伪谱等高线：

\[
\sigma_{\min}(zI-A)=\epsilon
\]

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

### 3. 检查数据集

```bash
python src/data/dataset.py --data-dir data/my_data
```

当前控制器输入维度是 `8`。

### 4. 训练控制器

```bash
python scripts/train_from_dataset.py \
    --data-dir data/my_data \
    --experiment-name my_model \
    --epochs 50 \
    --batch-size 256 \
    --device cuda
```

### 5. 对自己的矩阵做追踪

```bash
python scripts/run_tracking.py \
    --matrix-path path/to/matrix.npy \
    --checkpoint models/my_model/best_model.pt \
    --plot-out results/trajectory.png \
    --result-out results/trajectory.json \
    --epsilon 0.1
```

### 6. 与传统 Newton predictor-corrector 做对比

```bash
python scripts/benchmark_nn_vs_newton.py \
    --checkpoint models/my_model/best_model.pt \
    --output-dir results/nn_vs_newton
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
├── NN_8维输入梳理.md
├── DATASET_EXPLANATION.md
└── 算法完整流程与单步计算全过程.md
```

---

## 当前算法结构

### 1. 推理主链

推理时真正运行的是 `src/core/contour_tracker.py` 里的 tangent tracker：

- 当前状态是 `(z, u, v)`
- 切向方向由 `v^*u` 的白盒几何公式直接给出
- 网络只决定 `ds`
- tracker 负责近似接受、延迟投影、局部投影和必要时的精确刷新

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

- 专家推进：full triplet RK4 + 局部/全局投影
- 标签：只保留 `ds_expert`
- 增强：`DAgger`

---

## 为什么这个版本仍然有效

一个容易产生误解的点是：

> 传统算法慢，是因为要频繁做 SVD；而切向公式本身很便宜。你的方法如果还是会做 SVD，为什么还能更快？

关键不在于“把单次 SVD 变快”，而在于**减少昂贵操作出现的频率，并减少总 rollout 步数**。

当前实现真正节省时间的地方有三层：

1. **方向本来就便宜**
   - 切向方向由当前 `(u,v)` 直接给出
   - 这部分不需要学习

2. **网络学习的是步长控制**
   - 平滑区域走大步
   - 风险区域自动保守
   - 同一条 contour 往往能用更少步数走完

3. **tracker 会尽量推迟昂贵刷新**
   - 先尝试近似 triplet 判定
   - 必要时才做 exact SVD 或 projection
   - 因而总 wall-clock 可以下降

所以收益来源不是：

- “网络替代了 SVD”

而是：

- “网络减少了你不得不做昂贵校正的频率，并减少了总步数”
