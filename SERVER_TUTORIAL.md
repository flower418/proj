# 服务器操作保姆级教程

## 目录

1. [环境准备](#1-环境准备)
2. [生成数据](#2-生成数据)
3. [训练模型](#3-训练模型)
4. [推理可视化](#4-推理可视化)
5. [常见问题](#5-常见问题)

---

## 1. 环境准备

### 1.1 创建虚拟环境

```bash
conda create -n pseudospectrum python=3.10 -y
conda activate pseudospectrum
```

### 1.2 安装依赖

```bash
pip install -r requirements.txt
```

### 1.3 验证安装

```bash
pytest tests/ -v
```

**预期**：所有测试通过（约 10 个 PASSED）

---

## 2. 生成数据

### 2.1 快速测试（5 分钟，验证流程）

```bash
python scripts/generate_large_dataset.py \
    --target-samples 1000 \
    --matrix-sizes 20 30 \
    --trajectories-per-type 2 \
    --max-steps 50 \
    --output-dir data/test_1k \
    --seed 0
```

**输出**：`data/test_1k/dataset_full.npz` (约 1000 样本)

### 2.2 生产级数据（推荐，2-4 小时）

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

**输出**：`data/prod_50k/dataset_full.npz` (约 5 万样本)

### 2.3 大规模数据（10 万+，4-8 小时）

```bash
python scripts/generate_large_dataset.py \
    --target-samples 100000 \
    --matrix-sizes 30 50 80 100 \
    --trajectories-per-type 10 \
    --max-steps 300 \
    --dagger-factor 2 \
    --output-dir data/large_100k \
    --seed 0
```

**输出**：`data/large_100k/dataset_full.npz` (约 10 万样本)

### 2.4 检查数据

```bash
python src/data/dataset.py --data-dir data/prod_50k
```

**预期输出**：
```
Train: 40,000 samples (80%)
Val: 5,000 samples (10%)
Test: 5,000 samples (10%)
重启样本：4,000 (8.0%)
```

---

## 3. 训练模型

### 3.1 使用生成的数据训练

```bash
python scripts/train_from_dataset.py \
    --data-dir data/prod_50k \
    --experiment-name my_model \
    --epochs 50 \
    --batch-size 256 \
    --device cuda
```

**参数说明**：
- `--data-dir`: 步骤 2 生成的数据目录
- `--experiment-name`: 模型名称（自定义）
- `--epochs`: 训练轮数（50-100）
- `--batch-size`: 批大小（GPU 内存够就用 256）
- `--device`: `cuda` 或 `cpu`

**输出**：
- `models/my_model/best_model.pt` (最佳模型)
- `logs/my_model/` (TensorBoard 日志)

### 3.2 监控训练

打开新终端：
```bash
tensorboard --logdir=logs/ --port 6006
```

浏览器访问：`http://<服务器 IP>:6006`

### 3.3 训练完成标志

```
早停触发 (patience=15)
epochs_run=35 final_val=0.123456 records=50000
```

---

## 4. 推理可视化

### 4.1 使用训练好的模型推理

```bash
python scripts/run_tracking.py \
    --matrix-size 50 \
    --checkpoint models/my_model/best_model.pt \
    --plot-out results/my_trajectory.png \
    --max-steps 100
```

**输出**：`results/my_trajectory.png` (轨迹图)

### 4.2 评估模型性能

```bash
python scripts/evaluate.py \
    --checkpoint models/my_model/best_model.pt \
    --data-dir data/prod_50k \
    --split test \
    --metrics-out models/my_model_metrics.json
```

**输出**：`models/my_model_metrics.json`

**关键指标**：
- `accuracy`: 重启决策准确率 (>0.85 为优)
- `f1`: F1 分数 (>0.75 为优)
- `step_size_r2`: 步长预测 R² (>0.6 为优)

---

## 5. 常见问题

### Q1: 生成数据太慢

**解决方案**：减少矩阵大小和样本数
```bash
--matrix-sizes 20 30      # 只用小矩阵
--target-samples 5000     # 减少样本
```

### Q2: CUDA Out of Memory

**解决方案**：减小批大小
```bash
--batch-size 128  # 或 64
```

### Q3: 训练 Loss 不下降

**检查**：
1. 数据是否正常生成（步骤 2.4）
2. 学习率是否合适（默认 1e-3）
3. 重启样本比例是否在 5-15%

### Q4: 使用 1024×1024 大矩阵

```bash
python scripts/generate_large_dataset.py \
    --target-samples 1000 \
    --matrix-sizes 1024 \
    --trajectories-per-type 2 \
    --max-steps 30 \
    --output-dir data/test_1024
```

**注意**：1024×1024 矩阵生成很慢，建议先用小矩阵测试流程。

---

## 完整流程示例

### 快速验证（30 分钟）

```bash
# 1. 生成小数据集
python scripts/generate_large_dataset.py \
    --target-samples 1000 \
    --matrix-sizes 20 30 \
    --output-dir data/quick \
    --seed 0

# 2. 训练
python scripts/train_from_dataset.py \
    --data-dir data/quick \
    --experiment-name quick \
    --epochs 10 \
    --device cpu

# 3. 推理
python scripts/run_tracking.py \
    --matrix-size 20 \
    --checkpoint models/quick/best_model.pt \
    --plot-out results/quick.png
```

### 生产流程（6 小时）

```bash
# 1. 生成数据（2-4 小时）
python scripts/generate_large_dataset.py \
    --target-samples 50000 \
    --matrix-sizes 30 50 80 \
    --output-dir data/prod \
    --seed 0

# 2. 训练（2-3 小时）
python scripts/train_from_dataset.py \
    --data-dir data/prod \
    --experiment-name prod \
    --epochs 50 \
    --device cuda

# 3. 评估
python scripts/evaluate.py \
    --checkpoint models/prod/best_model.pt \
    --data-dir data/prod \
    --metrics-out models/prod_metrics.json

# 4. 可视化
python scripts/run_tracking.py \
    --matrix-size 50 \
    --checkpoint models/prod/best_model.pt \
    --plot-out results/prod.png
```

---

## 文件结构

```
proj/
├── scripts/
│   ├── generate_large_dataset.py  # 数据生成
│   ├── train_from_dataset.py      # 训练
│   ├── evaluate.py                # 评估
│   └── run_tracking.py            # 推理
├── src/
│   └── data/dataset.py            # 数据加载
├── data/
│   ├── test_1k/                   # 小数据集
│   ├── prod_50k/                  # 生产数据
│   └── large_100k/                # 大规模数据
├── models/
│   └── my_model/
│       └── best_model.pt          # 训练好的模型
├── logs/
│   └── my_model/                  # TensorBoard 日志
└── results/
    └── my_trajectory.png          # 推理结果图
```

---

## 命令速查表

| 目的 | 命令 |
|------|------|
| 生成数据 | `python scripts/generate_large_dataset.py --target-samples 50000 --output-dir data/my_data` |
| 检查数据 | `python src/data/dataset.py --data-dir data/my_data` |
| 训练 | `python scripts/train_from_dataset.py --data-dir data/my_data --experiment-name my_model` |
| 推理 | `python scripts/run_tracking.py --checkpoint models/my_model/best_model.pt --plot-out results/out.png` |
| 评估 | `python scripts/evaluate.py --checkpoint models/my_model/best_model.pt --data-dir data/my_data` |
| TensorBoard | `tensorboard --logdir=logs/` |

---

有问题？查看 `DATASET_EXPLANATION.md` 了解数据生成细节。
