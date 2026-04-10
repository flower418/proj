# 服务器实验教程

这份教程只保留当前真实可用的主流程：

1. 建环境
2. 生成大规模离线数据
3. 检查数据
4. 训练模型
5. 跑 NN vs Newton 对比
6. 对自己的矩阵做推理

## 1. 建环境

```bash
conda create -n pseudospectrum python=3.10 -y
conda activate pseudospectrum
pip install -r requirements.txt
```

可选检查：

```bash
python -m pytest tests -v
```

## 2. 生成大规模数据

现在训练数据统一使用：

- `scripts/generate_large_dataset.py`
- 随机矩阵类型
- 随机复平面点
- `epsilon = sigma_min(zI - A)`

当前默认支持的矩阵类型为：

- `random_complex`
- `random_hermitian`
- `random_real`
- `ill_conditioned`
- `random_normal`
- `banded_nonnormal`
- `low_rank_plus_noise`
- `jordan_perturbed`
- `block_structured`

### 正式实验示例

如果你要做正式训练数据生成，可以从下面这条开始：

```bash
python -u scripts/generate_large_dataset.py \
    --target-samples 1000000 \
    --matrix-sizes 30 50 80 100 128 \
    --trajectories-per-type 6 \
    --max-steps 240 \
    --dagger-factor 2 \
    --output-dir data/prod1m \
    --save-every 50000 \
    --seed 0
```

说明：

- `python -u` 很重要，服务器上可以持续看到进度
- `target_samples` 是下限，不是严格上限
- 脚本会随机抽矩阵类型，并循环你给出的尺寸列表
- 每到 `save_every` 的增量会保存一次 partial 数据

### 生成后会得到什么

目录里至少会有：

- `dataset_full.npz`
- `dataset_full_splits.npz`
- `dataset_splits.npz`
- `dataset_stats.json`
- `logs/.../run_config.json`
- `logs/.../progress.jsonl`
- `logs/.../generation_summary.json`

如果中途触发阶段性保存，还会有：

- `partial_<N>.npz`
- `partial_<N>_splits.npz`

## 3. 检查数据

```bash
python src/data/dataset.py --data-dir data/prod1m
```

你会看到：

- 数据文件路径
- 划分文件路径
- 总样本数
- 特征维度
- 重启样本比例
- train / val / test 样本数

注意：

- 当前特征维度是 `14`
- 划分优先按 `trajectory_id` 分组，而不是纯样本级随机切分

## 4. 训练模型

当前训练只使用 `configs/default.yaml`。
不再有单独的 `configs/training.yaml`。

```bash
python scripts/train_from_dataset.py \
    --data-dir data/prod1m \
    --experiment-name prod1m_model \
    --epochs 80 \
    --batch-size 256 \
    --learning-rate 3e-4 \
    --weight-decay 1e-5 \
    --hidden-dims 128 128 64 \
    --activation silu \
    --dropout 0.05 \
    --head-hidden-dim 64 \
    --step-size-min 1e-6 \
    --step-size-max 1e-1 \
    --lambda-step 1.0 \
    --lambda-restart 3.0 \
    --gradient-clip-norm 1.0
```

输出文件：

- `models/prod1m_model/best_model.pt`
- `models/prod1m_model/training_history.json`
- `models/prod1m_model/test_metrics.json`
- `logs/prod1m_model/training_summary.png`

## 5. 跑 NN vs Newton 对比

```bash
python scripts/benchmark_nn_vs_newton.py \
    --checkpoint models/prod1m_model/best_model.pt \
    --matrix-size 50 \
    --seed 0 \
    --max-steps 16000 \
    --output-dir results/bench_seed0
```

这个 benchmark 默认串行执行，先跑 `NN + ODE`，再跑 `Newton PC`。这样 `NN` 这一条会独占 CPU，单路速度最快。

输出文件：

- `results/bench_seed0/comparison_plot.png`
- `results/bench_seed0/comparison_summary.json`
- `results/bench_seed0/trajectories.npz`
- `results/bench_seed0/random_matrix.npy`
- `results/bench_seed0/logs/.../nn/summary.json`

benchmark 不再额外写 baseline summary 或 step 日志，只保留 NN 的 summary。

## 6. 对自己的矩阵做推理

```bash
python scripts/run_tracking.py \
    --matrix-path path/to/matrix.npy \
    --checkpoint models/prod1m_model/best_model.pt \
    --epsilon 0.1 \
    --plot-out results/my_matrix/tracked_contour.png \
    --result-out results/my_matrix/tracking_summary.json
```

如果你不提供 `--checkpoint`，它会退化成固定步长 ODE 跟踪。
