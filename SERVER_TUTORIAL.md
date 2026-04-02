# 服务器训练与推理教程

这份教程按当前代码实际行为整理，适合你已经准备好一份 `prod50k` 数据集、准备上服务器长时间跑训练的场景。

下面默认你的数据目录是 `data/prod_50k`。如果你实际目录叫 `data/prod50k`，把命令里的路径整体替换掉即可。

## 目录

1. [先确认数据集](#1-先确认数据集)
2. [环境准备](#2-环境准备)
3. [开始训练](#3-开始训练)
4. [评估训练结果](#4-评估训练结果)
5. [怎么做推理](#5-怎么做推理)
6. [推理效果到底是什么](#6-推理效果到底是什么)
7. [随机演示模式](#7-随机演示模式)
8. [常见问题](#8-常见问题)
9. [你的最短可执行流程](#9-你的最短可执行流程)

---

## 1. 先确认数据集

训练脚本 `scripts/train_from_dataset.py` 读取的是一个目录，其中：

- `dataset_full.npz` 是必需的
- `dataset_splits.npz` 最好有，但没有也可以自动生成

如果你的老数据只有 `dataset_full_splits.npz`，当前代码也会自动兼容并转换，不需要手工改名。

### 1.1 检查数据

```bash
python src/data/dataset.py --data-dir data/prod_50k
```

你应该至少看到：

```text
总样本：接近 50,000
重启样本：xxxx (x.xx%)

Train: 40,000
Val: 5,000
Test: 5,000
```

`Train / Val / Test` 一般会接近 `80% / 10% / 10%`，不要求精确到个位数完全一致。

如果这里都不对，先不要开训练。

---

## 2. 环境准备

### 2.1 创建环境

```bash
conda create -n pseudospectrum python=3.12 -y
conda activate pseudospectrum
```

### 2.2 安装依赖

```bash
pip install -r requirements.txt
```

### 2.3 检查 PyTorch 和 GPU

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
```

如果你打算用 GPU 训练，`cuda` 必须输出 `True`。当前训练脚本在你传 `--device cuda` 但 GPU 不可用时，会直接报错退出，不会偷偷回退到 CPU。

### 2.4 跑单测

```bash
pytest tests/ -v
```

---

## 3. 开始训练

### 3.1 先做一次 1 epoch 冒烟测试

正式长跑前，先确认训练、日志、checkpoint、测试评估整条链路都能通。

```bash
python scripts/train_from_dataset.py \
    --data-dir data/prod_50k \
    --experiment-name prod50k_smoke \
    --epochs 1 \
    --batch-size 64 \
    --device cuda
```

这一步主要看三件事：

- 能正常读到 train/val/test
- 能正常开始训练并打印 epoch 日志
- 结束后能写出 `best_model.pt`、训练图和 `test_metrics.json`

### 3.2 正式训练

如果冒烟测试没问题，再跑正式版本：

```bash
python scripts/train_from_dataset.py \
    --data-dir data/prod_50k \
    --experiment-name prod50k_v1 \
    --epochs 50 \
    --batch-size 256 \
    --device cuda
```

如果显存不够，把 `--batch-size` 降到 `128` 或 `64`。

### 3.3 训练脚本实际会做什么

当前 `train_from_dataset.py` 会按下面流程执行：

1. 从 `data/prod_50k` 加载 `train/val/test`
2. 构建控制器网络
3. 用 `ReduceLROnPlateau` 调整学习率
4. 用验证集 loss 做 early stopping
5. 把验证集最优权重保存到 `best_model.pt`
6. 训练结束后，用最优权重在测试集上评估
7. 保存训练历史、测试指标和总图

### 3.4 训练时会看到什么

每个 epoch 会打印一行，形如：

```text
[Epoch 001] train_loss=... val_loss=... step=... restart=... acc=... f1=... lr=... |W|=... |G|=...
```

字段含义：

- `train_loss`: 训练集总 loss
- `val_loss`: 验证集总 loss
- `step`: 验证集步长回归 loss
- `restart`: 验证集重启分类 loss
- `acc`: 重启分类准确率
- `f1`: 重启分类 F1
- `lr`: 当前学习率

训练结束时会打印：

```text
训练完成！epochs_run=... final_val=...
模型保存在：models/prod50k_v1/best_model.pt
训练总图：logs/prod50k_v1/training_summary.png
```

### 3.5 训练产物在哪里

正式训练 `--experiment-name prod50k_v1` 之后，重点看这些文件：

- `models/prod50k_v1/best_model.pt`
- `models/prod50k_v1/training_history.json`
- `models/prod50k_v1/test_metrics.json`
- `logs/prod50k_v1/training_summary.png`
- `logs/prod50k_v1/history.json`
- `logs/prod50k_v1/config.json`

### 3.6 训练正常的最低标准

至少满足：

- loss 不是 `nan` 或 `inf`
- `models/prod50k_v1/best_model.pt` 确实生成了
- `logs/prod50k_v1/training_summary.png` 能打开
- `models/prod50k_v1/test_metrics.json` 能写出来

---

## 4. 评估训练结果

训练脚本结束时已经会自动做一次测试集评估，但你也可以单独再跑一遍：

```bash
python scripts/evaluate.py \
    --checkpoint models/prod50k_v1/best_model.pt \
    --data-dir data/prod_50k \
    --split test \
    --device cuda \
    --metrics-out models/prod50k_v1/test_metrics_rerun.json
```

输出是一个 JSON，常看这些字段：

- `loss`
- `step_loss`
- `restart_loss`
- `accuracy`
- `precision`
- `recall`
- `f1`
- `step_size_mae`
- `step_size_rmse`
- `step_size_r2`

这是“在数据集标注上的拟合情况”，不是最终几何轨迹质量的全部。真正推理时，还要看 contour 是否闭合、重启次数、轨迹是否稳定。

---

## 5. 怎么做推理

### 5.1 你要看的最终效果：随机矩阵 + 随机点 + 完整等高线

如果你要看的就是这一条完整链路：

1. 随机生成一个矩阵 `A`
2. 在这个矩阵对应的复平面区域里随机取一个点 `z_random`
3. 计算 `epsilon = sigma_min(z_random I - A)`
4. 以这个点定义的那条伪谱等高线为目标
5. 用 `NN + ODE` 从该点出发追踪出一条完整 contour

那就直接跑这个脚本：

```bash
python scripts/demo_random_inference.py \
    --checkpoint models/prod50k_v1/best_model.pt \
    --matrix-size 20 \
    --sample-mode point_sigma \
    --point-sampler spectral_box \
    --max-steps 200 \
    --max-attempts 32 \
    --require-closed \
    --output-dir results/random_demo
```

这条命令现在就是按你说的逻辑走的。

输出文件：

- `results/random_demo/random_matrix.npy`
- `results/random_demo/tracked_contour.png`
- `results/random_demo/tracking_summary.json`

其中 `tracking_summary.json` 里最关键的是：

- `algorithm`: 当前会写成 `nn_plus_ode`
- `random_point`: 随机取到的复平面点
- `sigma_at_random_point`: 这个点对应的伪谱值
- `epsilon`: 实际追踪的 level set
- `start_point`: 真正作为追踪起点的点
- `closed`: 是否闭合成功
- `closure_error`: 闭合误差

当你使用：

- `--sample-mode point_sigma`

时，`epsilon` 就是由 `random_point` 本身定义出来的，所以：

- `start_point = random_point`
- 追踪的就是“经过这个随机点的那条伪谱等高线”

这就是你说的“完整算法效果”。

### 5.2 如果你要换成真实矩阵推理，输入是什么

`scripts/run_tracking.py` 做的是“给定矩阵 A，在指定 `epsilon` 水平上追踪一条伪谱等高线分量”。

它需要一个矩阵文件：

- `.npy`
- 或 `.npz`

如果是 `.npz`，要么包含键 `A`，要么里面只有一个数组。

### 5.3 最推荐的推理方式：真实矩阵 + 自动选起点

```bash
python scripts/run_tracking.py \
    --matrix-path path/to/A.npy \
    --checkpoint models/prod50k_v1/best_model.pt \
    --epsilon 0.1 \
    --auto-start rightmost \
    --max-steps 200 \
    --plot-out results/A_rightmost.png \
    --result-out results/A_rightmost.json
```

这条命令的意思是：

- 读取你的真实矩阵 `A`
- 用训练好的模型做步长/重启控制
- 在 `epsilon = 0.1` 的那条伪谱边界上找一个自动起点
- 从这个起点开始追踪一条完整轮廓

`--auto-start` 可选：

- `rightmost`
- `leftmost`
- `topmost`
- `bottommost`

它不是“随便选一个点”，而是围绕某个极值特征值自动挑一条分量的起点。

### 5.4 手动指定一个复平面起始猜测

如果你已经知道大概想追哪一块区域，可以给一个复平面点：

```bash
python scripts/run_tracking.py \
    --matrix-path path/to/A.npy \
    --checkpoint models/prod50k_v1/best_model.pt \
    --epsilon 0.1 \
    --z0-real 0.5 \
    --z0-imag 0.2 \
    --max-steps 200 \
    --plot-out results/A_manual.png \
    --result-out results/A_manual.json
```

注意：

- 这里的 `z0-real` / `z0-imag` 只是“初始猜测”
- 它不要求你正好落在边界上
- 脚本会先把这个点投影到真实的 `sigma_min(zI-A)=epsilon` 边界，再开始追踪

### 5.5 不带 checkpoint 也能跑，但那不是神经网络推理

```bash
python scripts/run_tracking.py \
    --matrix-path path/to/A.npy \
    --epsilon 0.1 \
    --auto-start rightmost \
    --max-steps 200 \
    --plot-out results/A_baseline.png \
    --result-out results/A_baseline.json
```

这种情况下：

- 不会加载神经网络控制器
- 追踪器会退回到固定步长数值追踪

这个命令很适合拿来和带 `--checkpoint` 的结果做对比。

---

## 6. 推理效果到底是什么

这是最容易误解的地方。

### 6.1 模型推理不是什么

它不是：

- 输入矩阵后直接“画完整轮廓”的黑盒
- 只靠神经网络猜出整条曲线
- 一次运行把所有分量都画出来

### 6.2 模型推理实际做什么

模型只负责两件事：

1. 每一步建议多大的步长 `ds`
2. 每一步要不要做一次 SVD restart

真正沿着边界前进的，仍然是数值 ODE 追踪器。

也就是说，推理阶段仍然是“数值算法 + 神经网络控制策略”的组合，不是完全用 NN 代替数学追踪。

### 6.3 一次推理的输出是什么

`run_tracking.py` 和 `demo_random_inference.py` 本质上都是一次只追踪一条连通等高线分量，并输出：

- 一张轨迹图
- 一个摘要 JSON

如果你跑的是 `run_tracking.py`，摘要 JSON 里最重要的字段：

- `closed`: 是否检测到闭合
- `closure_error`: 终点和起点的距离，越小越好
- `tracked_points`: 一共走了多少个点
- `num_restarts`: 发生了多少次 restart
- `path_length`: 轨迹总长度
- `winding_angle`: 绕行角度，通常完整绕一圈时绝对值会接近 `2π`
- `start_mode`: 是自动起点还是手动起点

如果你跑的是 `demo_random_inference.py`，还要额外看：

- `closed`
- `closure_error`
- `tracked_points`
- `num_restarts`
- `algorithm`
- `random_point`
- `sigma_at_random_point`
- `start_point`

### 6.4 怎么判断这次推理效果好不好

最低标准：

- `closed = true`
- `closure_error` 足够小
- 轨迹图上没有明显乱跳、折返、断裂

进一步可以比较：

- 同一矩阵、同一 `epsilon`、同一起点下
- 不带 `checkpoint` 的 baseline
- 带 `checkpoint` 的 neural controller

一般你关心的是：

- 是否更容易闭合
- 是否用更少的点完成一圈
- 是否 restart 更合理
- 轨迹是否更稳

### 6.5 为什么推理结果不等于“精度指标高就一定画得好”

因为训练集评估看的主要是：

- 步长拟合
- restart 分类

但真实推理还会受到这些因素影响：

- 起点选得对不对
- `epsilon` 是否合适
- 这条 contour 分量是否本来就难追
- `max_steps` 是否给够
- 数值投影和回退是否频繁触发

所以真正上线时，建议同时看：

- `evaluate.py` 的测试指标
- `run_tracking.py` 的真实轨迹结果

---

## 7. 随机演示模式

如果你只是想快速确认 `NN + ODE` 的完整链路能不能闭合一条随机 contour，可以跑：

```bash
python scripts/demo_random_inference.py \
    --checkpoint models/prod50k_v1/best_model.pt \
    --matrix-size 20 \
    --sample-mode point_sigma \
    --point-sampler spectral_box \
    --max-attempts 32 \
    --require-closed \
    --seed 0 \
    --output-dir results/random_demo
```

它会自动：

- 随机生成一个矩阵
- 在谱区域附近随机取一个复平面点
- 计算这个点对应的 `epsilon = sigma_min(zI-A)`
- 用这个点作为起点，运行 `NN + ODE` 联合追踪
- 输出图和摘要 JSON

结果文件在：

- `results/random_demo/random_matrix.npy`
- `results/random_demo/tracked_contour.png`
- `results/random_demo/tracking_summary.json`

这个模式最适合做两件事：

- 看完整算法是不是按你预期跑通了
- 看模型能不能从随机点出发闭合出一条完整 contour

---

## 8. 常见问题

### Q1: 训练一开跑就报 CUDA 不可用

说明你传了 `--device cuda`，但当前环境里 PyTorch 没看到 GPU。

先检查：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

如果是 `False`，先解决驱动 / CUDA / PyTorch 安装问题。

### Q2: 训练显存不够

先降 batch size：

```bash
--batch-size 128
```

或者：

```bash
--batch-size 64
```

### Q3: loss 不下降

先按顺序排查：

1. 数据集检查命令是否正常
2. 1 epoch 冒烟测试是否正常结束
3. `training_summary.png` 里 train/val 是否同步发散
4. 是否把 `epsilon`、数据目录、checkpoint 路径写错了

### Q4: 推理没有闭合

先试这些：

1. 把 `--max-steps` 提高到 `300` 或 `500`
2. 换一个 `--auto-start`
3. 改用手动 `--z0-real/--z0-imag`
4. 确认 `--epsilon` 真的是你要追踪的那条 level set

### Q5: 为什么只画出一条 contour，不是全部

因为当前追踪器的设计就是“一次运行追踪一条连通分量”。

如果你想看别的分量，需要：

- 换一个自动起点策略
- 或者换一个手动起始猜测点

---

## 9. 你的最短可执行流程

假设你已经有 `data/prod_50k`。

### 9.1 检查数据

```bash
python src/data/dataset.py --data-dir data/prod_50k
```

### 9.2 先跑 1 epoch 冒烟测试

```bash
python scripts/train_from_dataset.py \
    --data-dir data/prod_50k \
    --experiment-name prod50k_smoke \
    --epochs 1 \
    --batch-size 64 \
    --device cuda
```

### 9.3 正式训练

```bash
python scripts/train_from_dataset.py \
    --data-dir data/prod_50k \
    --experiment-name prod50k_v1 \
    --epochs 50 \
    --batch-size 256 \
    --device cuda
```

### 9.4 重跑测试集评估

```bash
python scripts/evaluate.py \
    --checkpoint models/prod50k_v1/best_model.pt \
    --data-dir data/prod_50k \
    --split test \
    --device cuda \
    --metrics-out models/prod50k_v1/test_metrics_rerun.json
```

### 9.5 对真实矩阵做推理

```bash
python scripts/run_tracking.py \
    --matrix-path path/to/A.npy \
    --checkpoint models/prod50k_v1/best_model.pt \
    --epsilon 0.1 \
    --auto-start rightmost \
    --max-steps 200 \
    --plot-out results/A_rightmost.png \
    --result-out results/A_rightmost.json
```

### 9.6 跑你要的最终效果

```bash
python scripts/demo_random_inference.py \
    --checkpoint models/prod50k_v1/best_model.pt \
    --matrix-size 20 \
    --sample-mode point_sigma \
    --point-sampler spectral_box \
    --max-steps 200 \
    --max-attempts 32 \
    --require-closed \
    --output-dir results/random_demo
```

### 9.7 看结果时只盯这几个点

- `models/prod50k_v1/best_model.pt` 是否存在
- `models/prod50k_v1/test_metrics.json` 或重跑评估 JSON 是否正常
- `results/A_rightmost.json` 里 `closed` 是否为 `true`
- `results/A_rightmost.json` 里 `closure_error` 是否够小
- `results/A_rightmost.png` 的轨迹是否平滑、闭合
- `results/random_demo/tracking_summary.json` 里 `algorithm` 是否为 `nn_plus_ode`
- `results/random_demo/tracking_summary.json` 里 `random_point`、`sigma_at_random_point`、`start_point` 是否都正常
- `results/random_demo/tracking_summary.json` 里 `closed` 是否为 `true`
- `results/random_demo/tracked_contour.png` 是否确实画出完整 contour

如果这些都正常，这一版模型就已经具备“拿真实矩阵跑一条 contour 分量追踪”的基本可用性。
