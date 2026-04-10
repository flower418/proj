# 数据集说明

这份文档只描述当前仓库里真实使用的离线数据生成流程：

- 脚本：`scripts/generate_large_dataset.py`
- 矩阵类型：内置支持集合中随机抽取
- 起点：在复平面随机取点
- `epsilon`：固定为 `sigma_min(zI - A)` 对应的随机点等高线

## 1. 生成命令

一个标准示例：

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

当前 CLI 只有这些参数：

- `--target-samples`
- `--matrix-sizes`
- `--trajectories-per-type`
- `--max-steps`
- `--dagger-factor`
- `--output-dir`
- `--save-every`
- `--seed`
- `--log-dir`

训练时直接使用 `configs/default.yaml` 里的 `training` 段。
不再有单独的 `configs/training.yaml`。

## 2. 矩阵类型

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

这些类型会写入输出数据里的 `matrix_type` 字段。

## 3. 随机点与 epsilon

当前数据生成不再暴露 `epsilon_mode`、`point_sampler`、`radius_range`、`box_padding`。

固定策略是：

1. 先生成随机矩阵 `A`
2. 根据 `A` 的谱中心和谱尺度，在复平面采一个随机点 `z_random`
3. 计算 `epsilon = sigma_min(z_random I - A)`
4. 从这条经过随机点的等高线出发生成专家轨迹

也就是说，生成任务始终和最终“随机点定义等高线”的推理任务一致。

## 4. 样本内容

`dataset_full.npz` 当前包含：

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

- `features` 是控制器输入
- `ds_expert` 是专家步长标签
- `y_restart` 是是否重启标签
- `source` 取值为 `expert` 或 `dagger`

## 5. 特征维度

当前特征维度是 `14`，不是旧版本文档里的 `7` 或 `10`。

组成是：

- 10 个基础特征
- 4 个上下文特征

## 6. 划分方式

脚本会同时写出：

- `dataset_full_splits.npz`
- `dataset_splits.npz`

划分优先按 `trajectory_id` 分组，避免同一条轨迹同时进入 train / val / test。

## 7. 输出文件

完整生成后，通常会看到：

- `dataset_full.npz`
- `dataset_full_splits.npz`
- `dataset_splits.npz`
- `dataset_stats.json`
- `logs/.../run_config.json`
- `logs/.../progress.jsonl`
- `logs/.../generation_summary.json`

如果中途触发增量保存，还会有：

- `partial_<N>.npz`
- `partial_<N>_splits.npz`

## 8. 检查数据

```bash
python src/data/dataset.py --data-dir data/prod100k
```

可以看到：

- 数据文件路径
- 划分文件路径
- 总样本数
- 特征维度
- 重启比例
- train / val / test 数量
