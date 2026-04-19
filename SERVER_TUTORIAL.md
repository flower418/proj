# 服务器实验教程

这份教程只保留当前仓库真实可用的主流程：

1. 建环境
2. 跑数据生成
3. 检查数据
4. 训练模型
5. 跑 benchmark
6. 对自己的矩阵做推理

---

## 1. 建环境

```bash
conda create -n proj python=3.12 -y
conda activate proj
pip install -r requirements.txt
```

建议先确认：

```bash
python -c "import numpy, scipy, torch; print('env ok')"
```

---

## 2. 跑数据生成

### 2.1 先在前台试一小轮

```bash
python -u scripts/generate_large_dataset.py \
    --target-samples 20000 \
    --matrix-sizes 30 50 80 \
    --trajectories-per-type 4 \
    --max-steps 200 \
    --dagger-factor 2 \
    --output-dir data/warmup20k \
    --save-every 5000 \
    --seed 0 \
    --log-dir logs/warmup20k
```

### 2.2 正式跑大数据

如果你想直接在服务器挂着跑，推荐用：

```bash
mkdir -p data/prod1m logs/prod1m
nohup python -u scripts/generate_large_dataset.py \
    --target-samples 1000000 \
    --matrix-sizes 30 50 80 100 128 \
    --trajectories-per-type 6 \
    --max-steps 240 \
    --dagger-factor 2 \
    --output-dir data/prod1m \
    --save-every 50000 \
    --seed 0 \
    --log-dir logs/prod1m \
    > logs/prod1m/generate.out 2>&1 &
```

查看进度：

```bash
tail -f logs/prod1m/generate.out
```

脚本还会额外写入：

- `logs/prod1m/.../run.log`
- `logs/prod1m/.../progress.jsonl`
- `logs/prod1m/.../generation_summary.json`

### 2.3 会产出什么

完整生成后，`data/prod1m/` 下通常会看到：

- `dataset_full.npz`
- `dataset_full_splits.npz`
- `dataset_splits.npz`
- `dataset_stats.json`
- `partial_<N>.npz`（如果触发了阶段性保存）
- `partial_<N>_splits.npz`（如果触发了阶段性保存）

---

## 3. 检查数据

```bash
python src/data/dataset.py --data-dir data/prod1m
```

你会看到：

- 数据文件路径
- 划分文件路径
- 总样本数
- 特征维度
- 步长最小值 / 最大值
- train / val / test 样本数

当前控制器输入维度是 **8**。

---

## 4. 训练模型

训练脚本使用 `configs/default.yaml` 的 `controller` 和 `training` 配置，并支持命令行覆盖。

```bash
python -u scripts/train_from_dataset.py \
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
    --step-size-min 1e-4 \
    --step-size-max 1e-1 \
    --lambda-step 1.0 \
    --gradient-clip-norm 1.0 \
    --device cuda
```

输出通常在：

- `models/prod1m_model/best_model.pt`
- `models/prod1m_model/training_history.json`
- `models/prod1m_model/test_metrics.json`
- `logs/prod1m_model/...`

---

## 5. 跑 NN tracker vs Newton benchmark

```bash
python scripts/benchmark_nn_vs_newton.py \
    --checkpoint models/prod1m_model/best_model.pt \
    --matrix-size 50 \
    --seed 0 \
    --max-steps 16000 \
    --output-dir results/bench_seed0
```

输出文件：

- `results/bench_seed0/comparison_plot.png`
- `results/bench_seed0/comparison_summary.json`
- `results/bench_seed0/trajectories.npz`
- `results/bench_seed0/random_matrix.npy`
- `results/bench_seed0/logs/.../nn/summary.json`

这份 benchmark 会在同一个随机矩阵、同一个随机点上比较：

- 当前 `NN tangent tracker`
- 传统 `Newton predictor-corrector`

---

## 6. 对自己的矩阵做推理

```bash
python scripts/run_tracking.py \
    --matrix-path path/to/matrix.npy \
    --checkpoint models/prod1m_model/best_model.pt \
    --epsilon 0.1 \
    --plot-out results/my_matrix/tracked_contour.png \
    --result-out results/my_matrix/tracking_summary.json
```

如果你想只跑白盒链路，也可以不提供 checkpoint：

```bash
python scripts/run_tracking.py \
    --matrix-path path/to/matrix.npy \
    --epsilon 0.1 \
    --plot-out results/my_matrix/fixed_step_tracker.png
```

这时脚本会使用**固定步长 tangent tracker**。

---

## 7. 服务器上最实用的三个命令

### 看数据生成日志

```bash
tail -f logs/prod1m/generate.out
```

### 看最新 run 目录

```bash
ls -lt logs/prod1m | head
```

### 看最新 summary

```bash
find logs/prod1m -name generation_summary.json | sort | tail -n 1 | xargs cat
```
