# 服务器实验教程

这份教程只保留现在真正要跑的主流程：

1. 建环境
2. 生成和任务一致的新数据
3. 检查数据
4. 训练模型
5. 跑 NN 推理
6. 跑 Newton baseline
7. 跑 NN vs Newton 对比

## 1. 建环境

```bash
conda create -n pseudospectrum python=3.10 -y
conda activate pseudospectrum
pip install -r requirements.txt
```

建议先验证基础依赖：

```bash
python -m pytest tests -v
```

如果服务器没有装 `pytest`，这一步可以先跳过。

## 2. 生成新数据

现在训练数据必须和最终任务一致：

- 随机生成矩阵 `A`
- 随机抽一个复平面点 `z_random`
- 计算 `epsilon = sigma_min(z_random I - A)`
- 用高精度教师方法跟踪这条等高线
- 把轨迹状态转成监督学习样本

这件事用新脚本 [scripts/generate_precise_dataset.py](/Users/ziwenxu/proj/scripts/generate_precise_dataset.py)。

### 正式生成

如果你要在 `50 / 80 / 100` 规模上开始正式生成，直接跑：

```bash
python -u scripts/generate_precise_dataset.py \
    --target-samples 50000 \
    --matrix-sizes 50 80 100 \
    --matrix-types complex real hermitian ill_conditioned \
    --trajectories-per-type 4 \
    --max-steps 4000 \
    --dagger-factor 1 \
    --sample-mode point_sigma \
    --point-samplers around_eigenvalue spectral_box \
    --radius-range 0.10 0.30 \
    --box-padding 0.10 \
    --expert-rtol 1e-9 \
    --expert-atol 1e-9 \
    --solver-tol 1e-10 \
    --drift-threshold 1e-5 \
    --output-dir data/prod50k_v2 \
    --seed 0
```

说明：

- `python -u` 很重要，这样服务器上会持续刷进度，不会看起来像“没显示”
- 这个脚本默认不画图，它是离线造训练数据，不是 demo
- 进度会按“每接受一条轨迹打印一行”的方式输出

### 生成后会得到什么

目录里至少会有：

- `dataset_full.npz`
- `dataset_full_splits.npz`
- `dataset_stats.json`
- `trajectory_metadata.jsonl`

其中：

- `dataset_full.npz` 是训练真正要吃的数据
- `trajectory_metadata.jsonl` 记录每条轨迹来自哪种矩阵、哪个随机点、对应哪个 `epsilon`

## 3. 检查数据

```bash
python src/data/dataset.py --data-dir data/prod50k_v2
```

你会看到：

- 数据文件路径
- 划分文件路径
- 总样本数
- 特征维度
- 重启样本比例
- train / val / test 样本数

注意：

- 现在新的特征维度是 `10`
- 这不是旧版的 `7` 维特征了

## 4. 训练模型

数据生成完之后，就直接用同一个离线训练脚本训练，不需要换训练框架。

```bash
python scripts/train_from_dataset.py \
    --data-dir data/prod50k_v2 \
    --experiment-name prod50k_v2_model \
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
    --gradient-clip-norm 1.0 \
    --device cuda
```

输出文件：

- `models/prod50k_v2_model/best_model.pt`
- `models/prod50k_v2_model/training_history.json`
- `models/prod50k_v2_model/test_metrics.json`
- `logs/prod50k_v2_model/training_summary.png`

说明：

- 脚本默认启用 early stopping
- 所以你设 `--epochs 80`，实际可能只跑到 40 多或 50 多轮
- 这不是报错，而是验证集 loss 连续若干轮没有提升

## 5. 跑 NN 推理

如果你要单独看你的 NN+ODE 效果，用这个脚本：

```bash
python scripts/demo_random_inference.py \
    --checkpoint models/prod50k_v2_model/best_model.pt \
    --matrix-size 50 \
    --seed 0 \
    --sample-mode point_sigma \
    --point-sampler around_eigenvalue \
    --output-dir results/nn_demo_seed0
```

输出文件：

- `results/nn_demo_seed0/random_matrix.npy`
- `results/nn_demo_seed0/tracked_contour.png`
- `results/nn_demo_seed0/tracking_summary.json`

这张图只画你的 NN+ODE 轨迹，不会混 baseline。

## 6. 跑 Newton Baseline

如果你要单独看经典 `Newton predictor-corrector` baseline，用这个脚本：

```bash
python scripts/run_newton_baseline.py \
    --matrix-size 50 \
    --seed 0 \
    --sample-mode point_sigma \
    --point-sampler around_eigenvalue \
    --output-dir results/newton_seed0
```

输出文件：

- `results/newton_seed0/random_matrix.npy`
- `results/newton_seed0/newton_baseline.png`
- `results/newton_seed0/trajectory.npz`
- `results/newton_seed0/summary.json`

这张图只画 Newton baseline，不会混 NN。

图上会直接标：

- `Time: ...s`

这里的时间口径是：

- 先算完 `epsilon`
- 再开始计时跟踪整条等高线

`summary.json` 里也会单独保存：

- `epsilon_compute_seconds`
- `elapsed_seconds`

## 7. 跑 NN vs Newton 对比

如果你要做同一矩阵、同一随机点、同一个 `epsilon` 下的公平比较，用这个脚本：

```bash
python scripts/benchmark_nn_vs_newton.py \
    --checkpoint models/prod50k_v2_model/best_model.pt \
    --matrix-size 50 \
    --seed 0 \
    --sample-mode point_sigma \
    --point-sampler around_eigenvalue \
    --max-steps 16000 \
    --output-dir results/bench_seed0 \
    --device cuda
```

这个脚本会严格做同一实例比较：

- 同一个随机矩阵
- 同一个随机点
- 同一个 `epsilon`
- 先算完 `epsilon`
- 然后分别给 NN 和 Newton 单独计时

### 会生成哪些图

会生成三张图：

- `comparison_plot.png`
  这张是混合图，NN 和 Newton 画在同一张图上，方便直接对比轮廓差异
- `nn_only_plot.png`
  这张只画你的 NN+ODE，而且和混合图用的是同一矩阵、同一起点
- `newton_only_plot.png`
  这张只画 Newton baseline，而且和混合图用的是同一矩阵、同一起点

还会生成：

- `comparison_summary.json`
- `trajectories.npz`
- `random_matrix.npy`

### 图上会不会标时间

会。

- `comparison_plot.png` 会同时标
  - `NN + ODE: ...s`
  - `Newton PC: ...s`
  - 如果 reference 轨迹存在，还会标 `Reference: ...s`
- `nn_only_plot.png` 会标 `Time: ...s`
- `newton_only_plot.png` 会标 `Time: ...s`

### JSON 里会记录什么

`comparison_summary.json` 里会记录：

- `epsilon_compute_seconds`
- `nn_plus_ode.elapsed_seconds`
- `newton_predictor_corrector.elapsed_seconds`
- `closed`
- `closure_error`
- `winding_angle`
- `mean_sigma_error`
- `max_sigma_error`
- `nn_vs_newton` 的轮廓距离
- 如果 reference 成功闭合，还会有
  - `nn_vs_reference`
  - `newton_vs_reference`

## 8. 常见问题

### 8.1 benchmark 画的是一张图还是三张图

三张：

- 一张混合对比图
- 一张 NN 单独图
- 一张 Newton 单独图

### 8.2 只想看我的模型，不想看 baseline

跑：

```bash
python scripts/demo_random_inference.py \
    --checkpoint models/prod50k_v2_model/best_model.pt \
    --matrix-size 50 \
    --seed 0 \
    --sample-mode point_sigma \
    --point-sampler around_eigenvalue \
    --output-dir results/nn_demo_seed0
```

### 8.3 只想看 baseline，不想看 NN

跑：

```bash
python scripts/run_newton_baseline.py \
    --matrix-size 50 \
    --seed 0 \
    --sample-mode point_sigma \
    --point-sampler around_eigenvalue \
    --output-dir results/newton_seed0
```

### 8.4 数据生成太慢

先缩小：

```bash
--matrix-sizes 30 50
--target-samples 5000
--max-steps 1000
```

### 8.5 CUDA 不可用

把所有脚本里的：

```bash
--device cuda
```

改成：

```bash
--device cpu
```

### 8.6 推理没闭合

先看对应 summary JSON 里的：

- `closed`
- `closure_error`
- `winding_angle`
- `path_length`
- `tracked_points`

如果你只是先验证流程，优先把规模缩小：

```bash
--matrix-size 20
--max-steps 4000
```
