# 组会说明稿

## 1. 这个项目想解决什么问题

目标是追踪矩阵 `A` 的 `epsilon`-伪谱等高线，也就是在复平面里沿着
`sigma_min(zI-A)=epsilon` 这条曲线走一圈。

如果完全用传统数值方法，每走一步都要频繁做精确 SVD 或牛顿校正，代价很高。
所以这个项目的核心想法不是“让神经网络直接画整条轮廓”，而是：

- 用白盒几何/ODE 给出正确的局部运动方向
- 用神经网络只学习控制策略
- 具体来说，网络只负责两个决策：
  - 这一步走多大，也就是步长 `ds`
  - 这一步是否值得做一次精确重启

所以它本质上是一个“神经网络控制器 + 白盒追踪器”的混合方法。

## 2. 整体流程

当前主线流程很简单：

1. 随机生成矩阵和起点，做离线数据集
2. 用专家策略给每个状态打标签：`ds_expert` 和 `y_restart`
3. 训练一个轻量级 MLP 控制器
4. 推理时用网络给出局部控制信号，再交给追踪器执行
5. 用 Newton predictor-corrector 作为传统 baseline 对比

当前实际用到的主线脚本是：

- `scripts/generate_large_dataset.py`
- `scripts/train_from_dataset.py`
- `scripts/run_tracking.py`
- `scripts/benchmark_nn_vs_newton.py`

## 3. 数据集是怎么生成的

### 3.1 随机任务怎么定义

每个训练任务都不是手工给 `epsilon`，而是这样定义的：

1. 随机抽一个矩阵类型
2. 按给定尺寸生成矩阵 `A`
3. 在谱中心附近随机采一个复平面点 `z_random`
4. 计算
   `epsilon = sigma_min(z_random I - A)`
5. 把“经过这个随机点的等高线”当成当前追踪任务

所以当前数据集里的任务定义和实际 benchmark / inference 是一致的：
都是“随机点定义一条等高线”，而不是预先固定一批 `epsilon`。

当前内置的矩阵类型有 9 类：

- `random_complex`
- `random_hermitian`
- `random_real`
- `ill_conditioned`
- `random_normal`
- `banded_nonnormal`
- `low_rank_plus_noise`
- `jordan_perturbed`
- `block_structured`

### 3.2 专家标签怎么来

数据不是从网络 rollout 里抄出来的，而是由 `ExpertSolver` 生成。

专家的特点是：

- 用 full triplet RK4 推进 `(z,u,v)`
- 每一步后做精确校正，保证点回到等高线上
- 如果当前状态已经明显漂移，就直接打 `restart` 标签

专家对每个状态给出两个标签：

- `ds_expert`: 下一步建议步长
- `y_restart`: 当前是否应该重启

此外还会保存一些诊断量，比如：

- 残差
- `sigma` 偏差
- 是否发生了投影
- 路径长度、绕行角度等

### 3.3 为什么还要 DAgger

如果只在“很干净的专家轨迹”上训练，网络会学得太理想化。
一旦推理时状态有漂移，它可能就不会处理。

所以当前数据生成还会做一层 DAgger 风格增强：

- 对专家轨迹上的状态 `(z,u,v)` 做小扰动
- 把扰动后的状态重新投影/查询专家
- 记录专家在“坏状态”下的恢复动作

这样训练集不仅有标准轨迹，也有偏离轨道后的恢复样本。

### 3.4 数据集里存什么

当前主数据文件 `dataset_full.npz` 里最重要的字段是：

- `features`
- `ds_expert`
- `y_restart`
- `epsilon`
- `matrix_size`
- `matrix_type`
- `matrix_id`
- `trajectory_id`
- `source`

其中：

- `source=expert` 表示原始专家轨迹样本
- `source=dagger` 表示扰动增强样本

训练/验证/测试划分优先按 `trajectory_id` 分组，避免同一条轨迹同时落到不同 split。

## 4. 神经网络架构

当前网络是一个很直接的两头 MLP。

默认配置来自 `configs/default.yaml`：

- 输入维度：`14`
- 隐层：`[128, 128, 64]`
- 激活函数：`SiLU`
- 归一化：`LayerNorm`
- dropout：`0.05`
- head hidden dim：`64`

结构上分成三部分：

1. 一个共享编码器
2. 一个步长回归头
3. 一个重启概率头

也就是说，网络不会直接输出几何轨迹，而是输出控制信号。

## 5. 网络输入是什么

输入不是原始矩阵条目，也不是整个轨迹片段，而是一个维度无关的 14 维局部状态特征。

### 5.1 前 10 维基础特征

它们来自当前状态 `(z,u,v)`：

1. 当前近似奇异值和目标 `epsilon` 的偏差
2. `u` 的范数偏差
3. `v` 的范数偏差
4. 残差 `||Mv - epsilon u||`
5. `|u^*v|`
6. 当前相位相对上一步的变化
7. 上一次线性求解器迭代次数
8. `epsilon` 的尺度
9. 矩阵尺度
10. 当前点 `z` 相对矩阵尺度的位置

这些量都会做归一化，很多是 log 归一化。

### 5.2 后 4 维上下文特征

它们不是当前几何量，而是控制上下文：

11. 距离上次 restart 过了多少步
12. 上一步用了多大步长
13. 上一步是否做了 projection
14. 上一步是否做了 restart

所以这个网络本质上看到的是“当前局部几何 + 最近控制历史”。

## 6. 网络输出是什么

网络有两个输出：

### 6.1 步长头

输出下一步步长 `ds`。

实现上：

- 如果设置了 `step_size_max`，就把输出压到 `[step_size_min, step_size_max]`
- 当前默认上界是 `0.1`

### 6.2 重启头

输出一个 `restart_prob`，表示当前位置是否应该做一次精确重启。

需要强调的是：

- 网络原始只输出概率
- 当前主线推理并不是直接用 `0.5` 阈值硬判
- 实际上它先经过 `AdaptiveInferenceController` 再决定是否真的 restart

这个推理时控制器额外做了：

- restart hysteresis：要连续高风险才真的触发
- restart cooldown：避免连续重启
- stable growth：连续稳定时把步长慢慢放大
- projection-aware ceiling：如果最近投影太多，就把步长上限临时压低
- curvature penalty：转向太猛时也会压步长

所以推理时真正用的是：

- `NNController`: 给原始 `ds` 和 `restart_prob`
- `AdaptiveInferenceController`: 做推理期的稳定化包装

这里有一个非常关键的实现细节：

- 当前推理里的 `restart` 不是“停在原地不动”
- 它的真实含义是：先在当前 `z` 上做一次精确 SVD，刷新 `(u,v)`，然后立刻继续这一推进步

所以 `restart` 更准确地说是 **exact triplet refresh**。

## 7. 训练方式

### 7.1 训练目标

训练时把问题拆成两个子任务：

- 回归 `ds_expert`
- 分类 `y_restart`

### 7.2 损失函数

总损失是两部分加权和：

- step loss
- restart loss

当前实现里：

- 步长部分用 log-MSE
  - 只在 `y_restart=0` 的样本上算
  - 这样更适合处理跨数量级步长
- restart 部分用带 focal 因子的加权 BCE
  - 因为 restart 是稀有事件，类别不平衡明显

### 7.3 优化方式

当前默认训练配置：

- 优化器：`AdamW`
- 学习率调度：`ReduceLROnPlateau`
- early stopping：有
- gradient clipping：有
- 默认设备：`cpu`

训练时直接读取离线数据集，不做在线交互训练。

## 8. 当前推理算法是什么

这一节最重要。

需要先明确一个事实：

**当前主线推理并不是 full triplet ODE rollout。**

虽然代码里保留了对 `(z,u,v)` 做 RK4 / Heun 积分的能力，但当前真正跑 benchmark 和 `run_tracking` 时，走的是 `FAST_TANGENT_TRACKER_KWARGS` 这条快路径。

这意味着：

- `z` 按 ODE 切向推进
- `u,v` 在大多数步里不会做完整 ODE 积分传播
- `u,v` 主要通过“近似复用 + 必要时精确刷新/投影”来维护

换句话说，当前主线更接近：

**NN 控制步长 + 切向 ODE 推 `z` + 低频精确校正 `u,v`**

而不是：

**每一步都完整积分 `(z,u,v)`**

### 8.0 当前推理真正用到的推导共识

当前代码在推理时真正依赖的是下面几个结论：

1. 在等高线上，最小奇异值三元组满足
   - `(zI-A)v = epsilon u`
   - `(zI-A)^*u = epsilon v`
2. 记
   - `gamma = v^*u`
   则等高线切向可以写成
   - `dz/ds = i * gamma / |gamma|`
3. 局部法向修正也由 `gamma` 给出
   - 法向方向就是 `gamma / |gamma|`
4. 即使不重新做精确 SVD，也可以用当前携带的 `(u,v)` 在新点上估计
   - 近似奇异值
   - 残差
   - 离等高线的大致距离

当前主线高频使用的是第 2、3、4 条。

也就是说，当前 NN+ODE 的分工是：

- **ODE / 推导给方向**
- **NN 给控制信号**
- **projection / exact SVD 保证没有完全跑飞**

### 8.1 初始化

推理开始前先做一次精确 SVD：

- 给定起点 `z0`
- 得到精确的 `(u,v)`
- 把当前状态初始化为 `(z,u,v)`

这是整个追踪过程的精确起点。

### 8.2 每一步的主循环

当前主线每一步可以概括成下面 6 步：

1. 从当前 `(z,u,v)` 提取 14 维特征
2. `NNController` 预测原始 `ds` 和 `restart_prob`
3. `AdaptiveInferenceController` 决定这一步最终实际用的 `ds`，以及是否真的触发 `restart`
4. 如果触发 `restart`，就在当前 `z` 上做一次精确 SVD，刷新 `(u,v)`
5. 用刷新后的或原有的 `(u,v)` 计算切向
   - `dz/ds = i * (v^*u) / |v^*u|`
   然后得到
   - `z_candidate = z + ds * dz/ds`
6. 检查当前 triplet 在 `z_candidate` 上是否还可信，并据此决定：
   - 直接接受
   - 暂时带误差接受，推迟修正
   - 做局部法向 projection
   - 必要时做精确刷新或更保守的回退投影

走完以后再更新闭合检测、路径长度、绕行角等统计量。

### 8.3 ODE 在这里具体起什么作用

当前主线里，ODE 的主要作用是提供 **切向方向**：

`dz/ds = i * gamma / |gamma|`

其中 `gamma` 来自当前的奇异向量对。

这一步告诉我们：

- 在当前点上，沿等高线应该往哪个方向走

所以网络并不负责“决定方向”，方向是白盒几何给的。
网络只负责“走多远、何时精确纠偏”。

### 8.4 当前推理里 `u,v` 是怎么处理的

这是当前实现里最容易被误解的地方。

当前主线推理时：

- 候选点通常先只更新 `z`
- `u,v` 默认先沿用旧值
- 然后用这对旧 triplet 在新点上估计：
  - 近似奇异值
  - 残差
  - 需要多大 projection 才能回到等高线

如果这些近似量还好，就跳过精确刷新。
如果不好，再做局部 projection 或精确 SVD 刷新。

所以当前版本真正快的地方不是“神经网络前向传播很快”，而是：

- 少做 SVD
- 少做完整的 `u,v` 精确更新
- 能用近似 triplet 的地方就先近似

### 8.5 当前主线为什么比 full ODE 更快

因为当前 fast path 做了三件事：

1. 不对 `(u,v)` 每一步做 full RK4
2. 允许若干步连续跳过 exact triplet refresh
3. 允许轻微偏离时先延期 correction，而不是立刻精确处理

这也是为什么当前版本的推理明显快于传统 baseline。

### 8.6 把当前 NN+ODE 用一句话说清楚

当前版本最准确的一句话是：

> 网络不预测轨迹本身，只预测局部控制量；方向由伪谱流形的切向公式给出，位置 `z` 按切向 ODE 前进，`u,v` 尽量复用旧 triplet，只在误差累积到一定程度时再做 projection 或精确 SVD 刷新。
