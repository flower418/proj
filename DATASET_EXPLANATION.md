# 数据集说明

这份文档描述当前仓库真实使用的离线数据生成流程。

数据脚本是：`scripts/generate_large_dataset.py`

训练目标是一个**纯步长监督**问题：

- 输入：8 维控制特征 `features`
- 标签：`ds_expert`
- 数据来源：`expert` 主轨迹 + `dagger` 扰动恢复样本

---

## 1. 任务是怎么定义的

每条轨迹任务都按下面的方式生成：

1. 随机选矩阵类型并生成矩阵 `A`
2. 在谱中心附近随机采样复平面点 `z_random`
3. 计算
   \[
   \epsilon = \sigma_{\min}(z_{random}I-A)
   \]
4. 把经过 `z_random` 的这条 `epsilon` 等高线作为当前追踪任务

所以数据生成、训练目标和最终推理的任务定义是一致的：

- 随机矩阵
- 随机点决定 `epsilon`
- 追踪对应 contour

---

## 2. 矩阵类型

当前脚本内部会随机抽取以下矩阵类型：

- `random_complex`
- `random_hermitian`
- `random_real`
- `ill_conditioned`
- `random_normal`
- `banded_nonnormal`
- `low_rank_plus_noise`
- `jordan_perturbed`
- `block_structured`

这些类型会写入数据的 `matrix_type` 字段。

---

## 3. 数据是怎么生成出来的

### 3.1 专家层

`scripts/generate_large_dataset.py` 会先调用 `src/train/expert_solver.py`：

- 状态：`(z, u, v)`
- 推进方式：full triplet RK4
- 纠偏方式：local projection / radial projection
- 输出标签：`ds_expert`

专家的作用不是直接给轨迹点分类，而是回答：

> 当前这个状态下，下一步走多远最合适？

### 3.2 Teacher-forced tracker 层

数据脚本不会直接把专家内部状态写进数据集，而是再跑一层 teacher-forced rollout：

- 使用专家给出的 `ds_expert`
- 调用和推理时同构的 `ContourTracker._advance_step(...)`
- 记录真实 rollout 分布下的 `(features, ds_expert)`

这样数据分布更接近真实推理链路。

### 3.3 DAgger 层

`src/train/dagger_augmentation.py` 会对轨迹点做小扰动：

- 扰动 `z`
- 扰动 `u`
- 扰动 `v`
- 尽量投影回 contour 附近
- 再查询专家给恢复步长

生成的样本会标记为 `source = "dagger"`。

---

## 4. 样本里到底有什么

`dataset_full.npz` 当前包含：

- `features`
- `ds_expert`
- `epsilon`
- `matrix_size`
- `matrix_type`
- `matrix_id`
- `trajectory_id`
- `source`

字段解释：

- `features`：已经完成归一化的 8 维控制器输入，`float32`
- `ds_expert`：专家建议步长，`float32`
- `epsilon`：当前 contour 的层级
- `matrix_size`：矩阵维度
- `matrix_type`：矩阵类型
- `matrix_id`：矩阵实例编号
- `trajectory_id`：轨迹编号
- `source`：`expert` 或 `dagger`

---

## 5. 特征维度

当前特征维度是 **8**。

这些特征由 `src/nn/features.py` 和 `src/core/contour_tracker.py` 共同组装：

- 6 个几何/尺度特征
- 1 个历史步长特征
- 1 个历史投影标记特征

详细公式见：`NN_8维输入梳理.md`

---

## 6. 划分方式

脚本会写出：

- `dataset_full_splits.npz`
- `dataset_splits.npz`

默认优先按 `trajectory_id` 分组，再切 train / val / test，避免同一条轨迹同时进入多个划分。

如果总轨迹条数太少，才会退化成样本级随机切分。

---

## 7. 生成命令

```bash
python -u scripts/generate_large_dataset.py \
    --target-samples 100000 \
    --matrix-sizes 30 50 80 100 \
    --trajectories-per-type 5 \
    --max-steps 200 \
    --dagger-factor 2 \
    --output-dir data/prod100k \
    --save-every 10000 \
    --seed 0 \
    --log-dir logs/prod100k
```

CLI 参数：

- `--target-samples`
- `--matrix-sizes`
- `--trajectories-per-type`
- `--max-steps`
- `--dagger-factor`
- `--output-dir`
- `--save-every`
- `--seed`
- `--log-dir`

---

## 8. 输出文件

一次完整生成通常会得到：

- `dataset_full.npz`
- `dataset_full_splits.npz`
- `dataset_splits.npz`
- `dataset_stats.json`
- `logs/.../run_config.json`
- `logs/.../progress.jsonl`
- `logs/.../generation_summary.json`

如果过程中触发阶段性保存，还会出现：

- `partial_<N>.npz`
- `partial_<N>_splits.npz`

---

## 9. 检查数据

```bash
python src/data/dataset.py --data-dir data/prod100k
```

你会看到：

- 数据文件路径
- 划分文件路径
- 总样本数
- 特征维度
- 步长最小值 / 最大值
- train / val / test 数量
