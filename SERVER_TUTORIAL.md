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
    --output-dir data/prod_50k \
    --seed 0
```

生成完成后，目录里至少应有：

- `dataset_full.npz`
- `dataset_full_splits.npz`
- `dataset_stats.json`

## 3. 检查数据

```bash
python src/data/dataset.py --data-dir data/prod_50k
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
    --data-dir data/prod_50k \
    --experiment-name my_model \
    --epochs 50 \
    --batch-size 256 \
    --device cuda
```

输出文件：

- `models/my_model/best_model.pt`
- `models/my_model/training_history.json`
- `models/my_model/test_metrics.json`
- `logs/my_model/training_summary.png`

说明：

- `--epochs` 和 `--batch-size` 现在会正确覆盖配置文件
- 如果服务器没有可用 GPU，不要传 `--device cuda`

## 5. 评估模型

### 用离线数据集评估

```bash
python scripts/evaluate.py \
    --checkpoint models/my_model/best_model.pt \
    --data-dir data/prod_50k \
    --split test \
    --device cuda \
    --metrics-out models/my_model_eval.json
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
    --checkpoint models/my_model/best_model.pt \
    --device cuda \
    --matrix-size 16 \
    --max-steps 4000 \
    --metrics-out models/my_model_synth_eval.json
```

这个模式下，脚本会先把给定的复平面猜测点投影到 `epsilon` 等高线上，再做评估。

## 6. 从随机点定义一条等高线并推理

这是最适合快速目测效果的脚本：

```bash
python scripts/demo_random_inference.py \
    --checkpoint models/my_model/best_model.pt \
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
    --checkpoint models/my_model/best_model.pt \
    --epsilon 0.1 \
    --plot-out results/my_contour.png \
    --result-out results/my_contour.json
```

如果你想手动指定一个起点猜测：

```bash
python scripts/run_tracking.py \
    --matrix-path path/to/matrix.npy \
    --checkpoint models/my_model/best_model.pt \
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
python src/data/dataset.py --data-dir data/prod_50k
```

这一步不训练，不推理，只检查数据文件和划分文件是否齐全。
