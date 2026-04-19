# 组会说明稿（当前代码版）

## 1. 任务

我们要追踪矩阵 `A` 的 `epsilon`-伪谱等高线：

\[
\sigma_{\min}(zI-A)=\epsilon
\]

真正难的地方通常不是切向公式本身，而是：

- 精确 SVD / Newton 校正太频繁
- 步长过于保守，导致总步数过多

所以当前项目的核心设计是：

- **方向交给白盒几何**
- **步长交给神经网络**
- **执行稳定性交给 tracker**

---

## 2. 当前主线

整个仓库现在的主线可以概括成：

1. 生成离线数据
2. 训练步长控制器
3. 推理时让网络预测 `ds`
4. 用白盒 tangent tracker 执行 contour rollout

核心脚本是：

- `scripts/generate_large_dataset.py`
- `scripts/train_from_dataset.py`
- `scripts/run_tracking.py`
- `scripts/benchmark_nn_vs_newton.py`

---

## 3. 数据集怎么来

### 3.1 任务定义

每条任务都这样构造：

1. 随机生成矩阵 `A`
2. 随机采样复平面点 `z_random`
3. 计算 `epsilon = sigma_min(z_random I - A)`
4. 追踪经过这个点的 `epsilon` contour

所以训练和推理面对的是同一种任务分布。

### 3.2 标签定义

标签由 `ExpertSolver` 给出，只有一个：

- `ds_expert`

专家内部做的是：

- full triplet RK4 推进 `(z,u,v)`
- local projection / radial projection 保证落回 contour

### 3.3 分布对齐

数据脚本还会做两层处理：

- **teacher-forced tracker**：用和推理同构的 `ContourTracker` 记录状态
- **DAgger**：扰动轨迹点，再问专家恢复步长

这样训练数据更接近真实 rollout 分布。

---

## 4. 网络学什么

### 输入

当前输入是 8 维局部特征，分成三组：

- 当前 triplet 是否可靠
- 当前任务和位置的尺度信息
- 上一步的控制反馈

### 输出

输出只有一个标量：

- `ds`

也就是说，网络学的是：

> 在当前局部状态下，下一步应该走多远。

---

## 5. 推理时到底怎么跑

推理主链是 `src/core/contour_tracker.py` 里的 `ContourTracker`。

每一步大致分为：

1. 从 `(z,u,v,prev_ds,prev_applied_projection)` 提 8 维特征
2. MLP 预测 `ds`
3. `AdaptiveInferenceController` 做轻量自适应包装
4. 用白盒切向公式做 predictor
5. 先尝试 approximate triplet 快路径
6. 不够稳时再升级到 local projection、exact SVD、radial projection
7. 更新状态并判断 contour 是否闭合

切向方向公式是：

\[
\frac{dz}{ds}=i\frac{v^*u}{|v^*u|}
\]

所以网络不负责“朝哪走”，只负责“走多远”。

---

## 6. 为什么这套方法可能更快

一个常见问题是：

> 传统算法慢在 SVD，你现在也没有把 SVD 完全消掉，为什么还可能更快？

答案是：

### 6.1 更少的总步数

好的步长策略会让：

- 平滑区走大步
- 风险区自动保守

于是同一条 contour 通常用更少 accepted steps 走完。

### 6.2 更少的精确刷新

tracker 不是每步都做最贵的精确刷新，而是优先：

- 复用已有 triplet
- 走 approximate triplet 快路径
- 只在必要时才升级到 exact SVD

### 6.3 更少的无效迂回

更合适的步长还能减少：

- 投影触发频率
- contour 上的抖动
- 闭合前的冤枉路

所以收益不是：

- “把一次 SVD 算快了”

而是：

- “让昂贵步骤出现得更少”
- “让整条 rollout 更短”
- “让总 wall-clock 更优”

---

## 7. 一句话总结

当前仓库最准确的概括是：

> **用 8 维局部状态让网络预测步长，用白盒切向公式负责方向，用 tangent tracker 执行 contour 追踪，并只在必要时才升级到更贵的纠偏。**
