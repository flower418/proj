# 神经增强伪谱等高线追踪

目标任务是追踪矩阵 `A` 的 `epsilon`-伪谱等高线：

\[
\sigma_{\min}(zI-A)=\epsilon
\]

当前仓库的主线非常明确：

- **方向**：白盒几何直接给出切向方向
- **步长**：神经网络只预测下一步 `ds`
- **执行器**：`ContourTracker` 负责真正 rollout，并在需要时做投影或精确刷新

也就是说，当前实现是一个 **step-size controller + tangent tracker** 系统。

---

## 1. 当前推荐工作流

主流程只有三步：

1. `scripts/generate_large_dataset.py` 生成离线数据
2. `scripts/train_from_dataset.py` 训练步长控制器
3. `scripts/run_tracking.py` / `scripts/benchmark_nn_vs_newton.py` 做推理和对比

---

## 2. 快速开始

### 2.1 安装环境

```bash
conda create -n proj python=3.12 -y
conda activate proj
pip install -r requirements.txt
```

### 2.2 生成训练数据

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

### 2.3 检查数据集

```bash
python src/data/dataset.py --data-dir data/my_data
```

### 2.4 训练控制器

```bash
python scripts/train_from_dataset.py \
    --data-dir data/my_data \
    --experiment-name my_model \
    --epochs 50 \
    --batch-size 256 \
    --device cuda
```

### 2.5 对自己的矩阵做追踪

```bash
python scripts/run_tracking.py \
    --matrix-path path/to/matrix.npy \
    --checkpoint models/my_model/best_model.pt \
    --plot-out results/trajectory.png \
    --result-out results/trajectory.json \
    --epsilon 0.1
```

如果不提供 `--checkpoint`，`run_tracking.py` 会退化成**固定步长 tangent tracker**，仍然使用同一条白盒几何链路，只是步长由固定常数给出。

### 2.6 与传统 Newton predictor-corrector 做对比

```bash
python scripts/benchmark_nn_vs_newton.py \
    --checkpoint models/my_model/best_model.pt \
    --output-dir results/nn_vs_newton
```

---

## 3. 当前项目结构

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
├── idea.md
└── 算法完整流程与单步计算全过程.md
```

---

## 4. 当前算法结构

### 4.1 推理主链

真正跑推理的是 `src/core/contour_tracker.py` 里的 `ContourTracker`：

- 当前状态是 `(z, u, v, prev_ds, prev_applied_projection)`
- 切向方向由 `v^*u` 的白盒公式直接给出
- 网络只决定 `ds`
- tracker 负责：
  - 近似 triplet 快路径
  - deferred accept
  - local projection
  - exact SVD refresh
  - radial/global projection

### 4.2 控制器输入

控制器输入总共 **8 维**。

其中前 6 维来自局部几何，后 2 维来自控制上下文。注意：网络真正接收的是这些量的**归一化版本**，不是原始物理量本身。

详细见：`NN_8维输入梳理.md`

### 4.3 控制器输出

网络只输出一个量：

- `ds`：下一步步长

在默认配置下，`NNController` 会把输出限制在 `[step_size_min, step_size_max]` 区间内；推理时外面还有 `AdaptiveInferenceController` 做额外的动态 ceiling 控制。

### 4.4 专家数据

训练标签来自 `src/train/expert_solver.py`：

- 专家推进：full triplet RK4
- 纠偏：local projection / radial projection
- 标签：`ds_expert`
- 数据分布对齐：teacher-forced tracker + DAgger

---

## 5. 代码里一条完整链路是什么

### 数据生成链

- `scripts/generate_large_dataset.py`
- `src/train/expert_solver.py`
- `src/train/dagger_augmentation.py`

流程：

1. 随机生成矩阵 `A`
2. 随机采样复平面点 `z_random`
3. 计算 `epsilon = sigma_min(z_random I - A)`
4. 专家给出 `ds_expert`
5. teacher-forced tracker 用同构 rollout 记录 `(features, ds_expert)`
6. DAgger 再补充偏离状态的恢复样本

### 训练链

- `scripts/train_from_dataset.py`
- `src/nn/controller.py`
- `src/nn/loss.py`

流程：

1. 读入 8 维 `features`
2. MLP 输出 `ds_pred`
3. 用 `log(ds_pred)` 和 `log(ds_expert)` 做 MSE
4. 保存 checkpoint 和训练日志

### 推理链

- `scripts/run_tracking.py`
- `src/core/contour_tracker.py`
- `src/nn/inference_controller.py`

流程：

1. 初始化 `(z0, u0, v0)`
2. 提取 8 维特征
3. 网络预测 `ds`
4. `AdaptiveInferenceController` 做步长稳定化
5. `ContourTracker` 执行一小步
6. 更新路径几何量并判断闭合

---

## 6. 为什么这个版本仍然可能更快

传统算法慢，核心通常不在切向公式本身，而在**精确 SVD / Newton 校正触发得太频繁**。

当前实现的收益主要来自三点：

1. **方向本来就是便宜的白盒量**
   - 切向方向直接由当前 `(u,v)` 给出

2. **网络学习的是“什么时候敢走大步”**
   - 平滑区更激进
   - 风险区更保守
   - 同一条 contour 常常能用更少 accepted steps 走完

3. **tracker 会优先走便宜路径**
   - 先试 approximate triplet
   - 再试 deferred accept / local projection
   - 只有必要时才升级到 exact SVD 或 radial projection

所以收益不在于“把单次 SVD 变快”，而在于：

- **减少昂贵刷新出现的频率**
- **减少整条 rollout 的总步数**
- **降低整体 wall-clock**

---

## 7. 进一步阅读

- `NN_8维输入梳理.md`：8 维输入的精确定义与哪些量真正进入网络
- `DATASET_EXPLANATION.md`：数据生成、字段和切分方式
- `算法完整流程与单步计算全过程.md`：从代码出发梳理完整链路和单步细节
- `idea.md`：适合组会或口头汇报的简化说明稿
- `classical_code.md`：论文/传统 predictor-corrector 参考稿，不代表当前仓库实现
