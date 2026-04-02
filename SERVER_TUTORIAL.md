# 服务器实验教程

这份教程只保留服务器上真正需要的主流程：

1. 建环境
2. 生成数据
3. 检查数据
4. 训练模型
5. 评估模型
6. 跑推理

## 1. 建环境

```bash
conda create -n pseudospectrum python=3.10 -y
conda activate pseudospectrum
pip install -r requirements.txt
```

建议先验证基础依赖：

```bash
pytest tests -v
```

## 2. 生成数据

### 快速验证

```bash
python scripts/generate_large_dataset.py \
    --target-samples 1000 \
    --matrix-sizes 20 30 \
    --trajectories-per-type 2 \
    --max-steps 50 \
    --output-dir data/test_1k \
    --seed 0
```

### 正式训练数据

```bash
python scripts/generate_large_dataset.py \
    --target-samples 50000 \
    --matrix-sizes 30 50 80 \
    --trajectories-per-type 8 \
    --max-steps 200 \
    --dagger-factor 2 \
    --output-dir data/prod50k \
    --seed 0
```

生成完成后，目录里至少应有：

- `dataset_full.npz`
- `dataset_full_splits.npz`
- `dataset_stats.json`

如果你服务器上已经有现成数据 `data/prod50k`，下面所有训练和评估命令都直接把目录换成它，不需要重新生成。

## 3. 检查数据

```bash
python src/data/dataset.py --data-dir data/prod50k
```

你会看到：

- 数据文件路径
- 划分文件路径
- 总样本数
- 特征维度
- 重启样本比例
- train / val / test 样本数

## 4. 训练模型

主实验请用离线数据集训练脚本：

```bash
python scripts/train_from_dataset.py \
    --data-dir data/prod50k \
    --experiment-name prod50k_v2 \
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

- `models/prod50k_v2/best_model.pt`
- `models/prod50k_v2/training_history.json`
- `models/prod50k_v2/test_metrics.json`
- `logs/prod50k_v2/training_summary.png`

说明：

- `--epochs` 和 `--batch-size` 现在会正确覆盖配置文件
- 脚本默认启用 early stopping，所以你设 `--epochs 80`，实际可能只跑到 40 多或 50 多轮就停了
- 提前停下不是报错，而是验证集 loss 连续若干轮没有提升
- 如果验证集里的 `step_size_mae / rmse` 还是明显卡住，可以继续试 `--step-size-max 5e-2` 或直接跑下面的 sweep
- 如果服务器没有可用 GPU，不要传 `--device cuda`

### 批量试多组结构和超参数

如果你要在 `prod50k` 上自动试几版网络结构，直接跑：

```bash
python scripts/train_sweep.py \
    --data-dir data/prod50k \
    --device cuda \
    --preset-set quick \
    --experiment-prefix prod50k_search \
    --epochs 80 \
    --batch-size 256 \
    --num-workers 4
```

这个脚本会自动尝试多组：

- 隐层宽度
- `relu / gelu / silu`
- 不同 dropout
- `lambda_step / lambda_restart`
- 不同 step-size 输出范围

结果会写到：

- `models/prod50k_search/<trial-name>/best_model.pt`
- `logs/prod50k_search/<trial-name>/training_summary.png`
- `models/prod50k_search/sweep_summary.json`

如果你只是先做烟雾测试，建议先跑：

```bash
python scripts/train_sweep.py \
    --data-dir data/prod50k \
    --device cuda \
    --preset-set quick \
    --experiment-prefix prod50k_smoke \
    --epochs 20 \
    --batch-size 256 \
    --trial-limit 2 \
    --num-workers 4
```

## 5. 评估模型

### 用离线数据集评估

```bash
python scripts/evaluate.py \
    --checkpoint models/prod50k_v2/best_model.pt \
    --data-dir data/prod50k \
    --split test \
    --device cuda \
    --metrics-out models/prod50k_v2_eval.json
```

关键输出指标通常包括：

- `accuracy`
- `precision`
- `recall`
- `f1`
- `step_size_mae`
- `step_size_rmse`
- `step_size_r2`

### 用单个随机矩阵做轨迹级评估

```bash
python scripts/evaluate.py \
    --checkpoint models/prod50k_v2/best_model.pt \
    --device cuda \
    --matrix-size 16 \
    --max-steps 4000 \
    --metrics-out models/prod50k_v2_synth_eval.json
```

这个模式下，脚本会先把给定的复平面猜测点投影到 `epsilon` 等高线上，再做评估。

## 6. 从随机点定义一条等高线并推理

这是最适合快速目测效果的脚本：

```bash
python scripts/demo_random_inference.py \
    --checkpoint models/prod50k_v2/best_model.pt \
    --matrix-size 20 \
    --seed 0 \
    --output-dir results/random_demo
```

逻辑是：

- 随机生成矩阵 `A`
- 随机抽一个复平面点 `z_random`
- 计算 `epsilon = sigma_min(z_random I - A)`
- 从该点对应的等高线出发闭合追踪

输出文件：

- `results/random_demo/random_matrix.npy`
- `results/random_demo/tracked_contour.png`
- `results/random_demo/tracking_summary.json`

图里的标题和图例排版已经改过，默认会尽量放到图外上方，不再直接压住轨迹主体。

可选参数：

- `--point-sampler spectral_box`
- `--point-sampler around_eigenvalue`
- `--sample-mode point_sigma`
- `--sample-mode trained_epsilon`

默认 `--sample-mode point_sigma`，也就是“随机点定义等高线”。

## 7. 对自己的矩阵做推理

```bash
python scripts/run_tracking.py \
    --matrix-path path/to/matrix.npy \
    --checkpoint models/prod50k_v2/best_model.pt \
    --epsilon 0.1 \
    --plot-out results/my_contour.png \
    --result-out results/my_contour.json
```

如果你想手动指定一个起点猜测：

```bash
python scripts/run_tracking.py \
    --matrix-path path/to/matrix.npy \
    --checkpoint models/prod50k_v2/best_model.pt \
    --epsilon 0.1 \
    --z0-real 0.3 \
    --z0-imag -0.2 \
    --plot-out results/my_contour.png \
    --result-out results/my_contour.json
```

说明：

- 脚本会先把 `z0` 猜测点投影到真实等高线
- 如果你不传 `z0`，脚本会自动选起点
- 默认最大步数来自 `configs/default.yaml` 的 `tracker.max_steps=4000`

## 8. 常见问题

### 8.1 数据生成太慢

先缩小实验规模：

```bash
--matrix-sizes 20 30
--target-samples 5000
--max-steps 100
```

### 8.2 CUDA 不可用

把所有脚本里的 `--device cuda` 改成：

```bash
--device cpu
```

### 8.3 推理没闭合

先检查：

- `tracking_summary.json` 里的 `closed`
- `closure_error`
- `winding_angle`
- `num_restarts`
- `num_projections`

如果你只是想先验证流程，优先跑：

```bash
python scripts/demo_random_inference.py \
    --checkpoint models/my_model/best_model.pt \
    --matrix-size 12 \
    --seed 0 \
    --output-dir results/smoke_demo
```

### 8.4 只想快速检查数据目录是否正常

```bash
python src/data/dataset.py --data-dir data/prod50k
```

这一步不训练，不推理，只检查数据文件和划分文件是否齐全。
