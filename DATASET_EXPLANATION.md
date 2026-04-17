# 数据集说明

这份文档描述当前仓库真实使用的离线数据生成流程。

当前版本是**纯步长监督**：

- 脚本：`scripts/generate_large_dataset.py`
- 标签：只保留 `ds_expert`
- 控制器输入：8 维
- 不再包含 restart 标签

---

## 1. 生成命令

```bash
python -u scripts/generate_large_dataset.py \
    --target-samples 100000 \
    --matrix-sizes 30 50 80 100 \
    --trajectories-per-type 5 \
    --max-steps 200 \
    --dagger-factor 2 \
    --output-dir data/prod100k \
    --save-every 10000 \
    --seed 0
```

当前 CLI 参数：

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

## 2. 任务如何定义

固定策略如下：

1. 生成随机矩阵 `A`
2. 在谱中心附近随机采样复平面点 `z_random`
3. 计算 `epsilon = sigma_min(z_random I - A)`
4. 追踪经过该点的 `epsilon` 等高线

因此数据生成和最终推理任务是一致的，都是“随机点定义一条 contour”。

---

## 3. 矩阵类型

脚本内部会随机抽取以下矩阵类型：

- `random_complex`
- `random_hermitian`
- `random_real`
- `ill_conditioned`
- `random_normal`
- `banded_nonnormal`
- `low_rank_plus_noise`
- `jordan_perturbed`
- `block_structured`

这些类型会写入输出数据的 `matrix_type` 字段。

---

## 4. 样本内容

`dataset_full.npz` 当前包含：

- `features`
- `ds_expert`
- `epsilon`
- `matrix_size`
- `matrix_type`
- `matrix_id`
- `trajectory_id`
- `source`

其中：

- `features`：控制器输入
- `ds_expert`：专家建议步长
- `source`：`expert` 或 `dagger`

---

## 5. 特征维度

当前特征维度是 `8`，由两部分组成：

- 6 个基础几何特征
- 2 个控制上下文特征

详见 `NN_8维输入梳理.md`。

---

## 6. 专家与 teacher-forced 轨迹

当前数据生成分两层：

1. **专家层**
   - `ExpertSolver` 用 full triplet RK4 推进 `(z,u,v)`
   - 对候选点做局部投影或全局投影
   - 输出 `ds_expert`

2. **teacher-forced tracker 层**
   - `ContourTracker` 用和推理一致的 tangent tracker 执行这一步 `ds_expert`
   - 记录真实 rollout 分布下的 `(features, ds_expert)`

这样生成的数据更接近真实推理分布，而不是只来自理想化专家状态。

---

## 7. 划分方式

脚本会写出：

- `dataset_full_splits.npz`
- `dataset_splits.npz`

划分优先按 `trajectory_id` 分组，避免同一条轨迹同时进入 train / val / test。

---

## 8. 输出文件

完整生成后通常会看到：

- `dataset_full.npz`
- `dataset_full_splits.npz`
- `dataset_splits.npz`
- `dataset_stats.json`
- `logs/.../run_config.json`
- `logs/.../progress.jsonl`
- `logs/.../generation_summary.json`

如果中途触发增量保存，还会出现：

- `partial_<N>.npz`
- `partial_<N>_splits.npz`

---

## 9. 检查数据

```bash
python src/data/dataset.py --data-dir data/prod100k
```

可以看到：

- 数据文件路径
- 划分文件路径
- 总样本数
- 特征维度
- 步长最小值 / 最大值
- train / val / test 数量
