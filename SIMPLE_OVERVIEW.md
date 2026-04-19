# 最简说明

## 1. 任务

目标是追踪矩阵 `A` 的一条伪谱等高线：

\[
\sigma_{\min}(zI-A)=\epsilon
\]

方法不是让网络直接画曲线，而是只学一件事：**下一步走多大**。

```text
当前局部状态 -> NN 预测步长 ds -> 白盒 tracker 沿切向走一步 -> 必要时做纠偏
```

---

## 2. 数据

每条样本只有两部分：

- 输入：当前状态的 **6 维特征**
- 标签：专家步长 `ds_expert`

所以本质就是一步回归：

```text
features -> ds_expert
```

数据生成方式：

- 随机生成矩阵 `A`
- 默认 **75%** 从特征值附近采样起点
- 默认 **25%** 从谱区域随机采样起点
- 对每个起点先算 `epsilon = sigma_min(z0 I - A)`
- 再让 teacher 沿这条 contour 走一段
- 每一步只记录 `(features, ds_expert)`

当前保存到数据集里的核心数组只有：

- `features`
- `ds_expert`

另外用 `trajectory_id` 做按轨迹切分，避免同一条轨迹同时出现在 train/val/test。

---

## 3. 网络输入：6 维

当前真正送进 MLP 的 6 维输入是：

1. `|<u,(zI-A)v>| - epsilon|` 的近似 contour 误差
2. `||(zI-A)v - epsilon u||` 的 triplet 残差
3. `|u^* v|`
4. `|epsilon|`
5. 矩阵尺度 `||A||_F / sqrt(n)`
6. 上一步步长 `prev_ds`

其中：

- 前 5 维是当前局部几何状态
- 第 6 维是一步历史

已经删掉、不再使用的输入：

- 位置尺度 `|z| / matrix_scale`
- `prev_applied_projection`

所以现在没有“定义了但根本没进网络”的旧特征残留。

---

## 4. 输出

输出只有一个标量：

- `ds`：下一步步长

没有 restart，没有 backtrack，没有分类头，也没有别的分支。

---

## 5. 推理主链

推理时维护的状态是：

- `z`：当前位置
- `u, v`：当前最小奇异值对应的左右奇异向量
- `prev_ds`：上一步实际接受的步长

每一步计算流程：

### 第 1 步：提特征

从当前 `(z, u, v)` 计算 5 个几何特征，再拼上 `prev_ds`，得到 6 维输入。

### 第 2 步：网络预测步长

MLP 输出原始步长，再经过步长上下界裁剪，得到这一步候选 `ds`。

### 第 3 步：白盒切向预测

tracker 用

\[
\dot z \propto i \frac{v^*u}{|v^*u|}
\]

给出 contour 的切向方向，然后做一步 predictor：

```text
z_candidate = z + ds * tangent
```

### 第 4 步：先走 cheap path

tracker 优先尝试便宜链路：

- 沿用近似 triplet
- 检查近似 `sigma_error`
- 检查 triplet residual

如果这一步还足够可信，就直接接受，跳过 exact SVD。

### 第 5 步：必要时再升级

如果 cheap path 不够稳，才逐级升级：

1. local projection
2. exact SVD refresh
3. radial projection fallback

所以推理的真实结构是：

```text
6维特征 -> NN给步长 -> tangent predictor -> 尽量 cheap accept -> 必要时纠偏
```

---

## 6. 单步计算全过程

这里按代码链路讲一次完整单步。

### 6.1 当前状态

已知上一步接受后的状态：

- `z_k`
- `u_k, v_k`
- `prev_ds_k`

### 6.2 组装输入

构造

- `sigma_error_k`
- `triplet_residual_k`
- `|u_k^* v_k|`
- `|epsilon|`
- `matrix_scale`
- `prev_ds_k`

做对数归一化后，得到 6 维向量 `x_k`。

### 6.3 网络前向

网络计算：

```text
x_k -> MLP -> ds_raw -> clamp -> ds_k
```

得到候选步长 `ds_k`。

### 6.4 切向 predictor

用当前 `u_k, v_k` 计算切向方向 `t_k`，然后走：

```text
z_candidate = z_k + ds_k * t_k
```

### 6.5 近似可接受性判断

在 `z_candidate` 上，tracker 先不急着做 exact SVD，而是先看：

- 近似 `sigma` 离 `epsilon` 差多少
- 近似 residual 大不大
- 估计这次偏离 contour 的距离大不大

如果这些量都在容忍范围内，就直接接受这一步。

### 6.6 如果不够稳，再纠偏

如果不满足 cheap accept，才会继续：

- 先试局部法向投影
- 不行再做 exact SVD
- 还不行再做 radial projection

最后得到真正接受的下一个点：

- `z_{k+1}`
- `u_{k+1}, v_{k+1}`
- `accepted_ds`

### 6.7 更新历史量

然后更新：

- `prev_ds <- accepted_ds`
- 路径长度
- 绕行角
- 是否闭合

再进入下一步。

---

## 7. 训练主链

训练非常简单：

1. 读取数据集 `features, ds_expert`
2. 用 MLP 预测 `ds_pred`
3. 对 `ds_pred` 和 `ds_expert` 做回归损失
4. 在验证集早停
5. 保存 `best_model.pt`

没有 DAgger，没有在线更新，没有额外分支。

---

## 8. 为什么它比传统 Newton predictor-corrector 快

关键点不是“算步长本身便宜”。

因为无论是规则算步长还是网络算步长，这一部分本来都不是主耗时。

真正耗时的是：

- exact SVD
- Newton corrector
- line search shrink
- predictor halving
- 各种失败后重试

所以更准确地说：

> 你的方法更快，不是因为“预测步长”这个动作便宜，
> 而是因为它给出的步长更容易让轨迹落在 tracker 的 cheap accept 区域，
> 从而减少 exact SVD / projection / corrector 触发次数，同时减少总步数。

也就是说，省下来的 wall-clock 主要来自两件事：

### 8.1 总步数更少

同一条 contour，NN 往往能更快找到合适的步长区间，所以接受步数更少。

### 8.2 每个 accepted step 背后的隐藏代价更低

传统 Newton PC 虽然表面上一步也是一个 `ds`，但这一步背后常常伴随：

- 多次 halving
- 多次 corrector 迭代
- 多次 line search
- 多次 exact SVD

而你的链路里，很多步可以停在近似 triplet 的 cheap path，不需要升级到最贵分支。

所以真正快在：

```text
更少的 accepted steps
+ 更少的昂贵纠偏/刷新
```

不是快在“步长公式计算量”本身。

---

## 9. 当前保留的核心文件

最核心主链现在只剩这些：

- `scripts/generate_large_dataset.py`：生成数据
- `scripts/train_from_dataset.py`：训练
- `scripts/run_tracking.py`：推理
- `src/nn/features.py`：6 维特征
- `src/nn/controller.py`：步长网络
- `src/nn/inference_controller.py`：推理期步长包装
- `src/core/contour_tracker.py`：主 tracker
- `src/train/expert_solver.py`：teacher 步长生成

已经去掉或清掉的冗余：

- DAgger
- restart 分支
- backtrack 分支
- 无用 ODE 推理分支
- `training.noise_std`
- `prev_applied_projection`
- 位置尺度 `|z| / matrix_scale`
- `src/solvers/krylov.py`
- `__pycache__`

---

## 10. 最简一句话

可以把你的方法直接概括成：

> 用 6 维局部几何特征预测下一步步长，方向仍由白盒切向公式给出；推理时优先走 cheap tracker 路径，只在必要时才做更贵的 SVD / 投影纠偏，因此整体比传统 Newton predictor-corrector 更快。
