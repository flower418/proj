# NN 8 维输入梳理（当前代码版）

当前控制器真正吃到的是 **8 维输入**。

对应代码位置：

- `src/nn/features.py`
- `src/core/contour_tracker.py`
- `src/train/dagger_augmentation.py`

这里最重要的结论只有两条：

1. **网络实际输入就是这 8 维，没有别的隐藏输入**
2. **这 8 维全部会被用到，其他量最多只用于 tracker / 日志 / 闭合判断**

---

## 1. 归一化函数

前 7 维里的连续量都会经过 `_log_normalize(...)`：

\[
\mathrm{lognorm}(x;a,b)=\mathrm{clip}\left(2\cdot\frac{\log_{10}(\max(x,10^a))-a}{b-a}-1,-1,1\right)
\]

所以网络看到的并不是原始物理量，而是压到 `[-1, 1]` 附近的数值。

---

## 2. 当前真正送进网络的 8 维输入

| 维度 | 原始语义量 | 代码中的实际变换 | 是否进入网络 | 作用 |
|---|---|---|---|---|
| 1 | `|sigma_approx - epsilon|` | `lognorm(x; -12, 0)` | 是 | 当前点离 contour 的近似偏差 |
| 2 | `||Mv - epsilon u||` | `lognorm(x; -12, 0)` | 是 | 当前 triplet 残差 |
| 3 | `|u^*v|` | `lognorm(x; -6, 0)` | 是 | 几何病态程度 |
| 4 | `|epsilon|` | `lognorm(x; -12, 2)` | 是 | contour 层级尺度 |
| 5 | `matrix_scale = ||A||_F / sqrt(n)` | `lognorm(x; -6, 3)` | 是 | 矩阵整体尺度 |
| 6 | `|z| / matrix_scale` | `lognorm(x; -6, 3)` | 是 | 当前点的相对位置尺度 |
| 7 | `prev_ds` | `lognorm(x; -8, -1)` | 是 | 上一步的步长历史 |
| 8 | `prev_applied_projection` | `0` 或 `1` | 是 | 上一步是否发生投影 |

其中：

- `M = zI - A`
- `sigma_approx = |u^*Mv|`

---

## 3. 哪些量真的会影响 MLP

答案很直接：

- **上表 8 维，全部会进入 MLP**
- **没有第 9 维、第 10 维，也没有额外 head 的隐式输入**

`NNController` 的 `input_dim` 默认就是 `8`，`configs/default.yaml` 里也是 `8`。

---

## 4. 哪些量不进入网络，但仍然在全链路里有用

下面这些量在当前代码里确实会被用到，但**不是 MLP 输入**。

### 4.1 用于推理时的自适应包装

`AdaptiveInferenceController.observe_step(...)` 会用：

- `raw_sigma_error`
- `projection_distance`
- `tangent_turn`
- `applied_projection`
- `ds`

这些量用于动态收紧或放松步长 ceiling。

### 4.2 用于闭合判断

`ContourTracker.check_closure(...)` 会用：

- `path_length`
- `max_distance_from_start`
- `winding_angle`
- `last_step_size`
- `z_prev`

这些量只服务于“这条 contour 是否已经闭合”。

### 4.3 用于日志和诊断

step callback / JSON 日志里还会记录：

- `raw_ds`
- `sigma`
- `projection_mode`
- `triplet_refresh_mode`
- `triplet_residual`
- `steps_since_exact_triplet_refresh`

这些也不进入网络。

---

## 5. 这 8 维怎么理解

可以把它们分成三组：

### A. 当前状态是否可靠

- `|sigma_approx - epsilon|`
- `||Mv - epsilon u||`
- `|u^*v|`

### B. 当前任务和位置处在什么尺度上

- `|epsilon|`
- `matrix_scale`
- `|z| / matrix_scale`

### C. 上一步控制反馈是什么

- `prev_ds`
- `prev_applied_projection`

所以网络学到的并不是“整个矩阵长什么样”，而是：

> 当前局部状态稳不稳、尺度大不大、上一步是不是已经走得太激进。

---

## 6. 一句话总结

当前代码里：

- **真正喂给网络的只有这 8 维**
- **这 8 维全部有效**
- **其他量要么用于 tracker 的后处理，要么用于闭合判断和日志，不会进入 MLP**
