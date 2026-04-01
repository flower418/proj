# 数据集详细说明文档

## 1. 矩阵大小规定

### ✅ 可以任意指定

矩阵大小**没有固定限制**，可以指定为任意值：

```bash
# 小型矩阵
--matrix-sizes 16 32 64

# 中型矩阵
--matrix-sizes 50 80 100

# 大型矩阵
--matrix-sizes 256 512 1024

# 混合尺寸
--matrix-sizes 30 50 100 256 512 1024
```

### 不同规模矩阵的对比

| 矩阵大小 | 内存占用 | 生成速度 | 用途 |
|----------|----------|----------|------|
| 16×16 | ~4 KB | 极快 (<1 秒/步) | 快速测试、调试 |
| 50×50 | ~40 KB | 快 (~5 秒/步) | 中等规模实验 |
| 100×100 | ~160 KB | 中等 (~20 秒/步) | 生产级训练 |
| 256×256 | ~1 MB | 较慢 (~2 分钟/步) | 大规模测试 |
| 512×512 | ~4 MB | 慢 (~10 分钟/步) | 压力测试 |
| 1024×1024 | ~16 MB | 很慢 (~1 小时/步) | 超大规模场景 |

### 内存占用计算

对于 n×n 复矩阵：
- 矩阵 A: `n² × 16 bytes` (复数 128 位)
- 中间矩阵 M = zI - A: `n² × 16 bytes`
- SVD 计算：`O(n³)` 时间复杂度

**示例**：
```
1024×1024 复矩阵:
- 矩阵存储：1024² × 16 = 16 MB
- SVD 一次：约 1024³ ≈ 10⁹ 次操作，约 10-60 秒
```

### 推荐配置

根据你的需求选择：

```bash
# 快速验证（几分钟）
--matrix-sizes 16 32 50

# 生产训练（几小时）
--matrix-sizes 50 80 100

# 超大规模研究（几天）
--matrix-sizes 100 256 512

# 包含超大矩阵
--matrix-sizes 50 100 256 512 1024
```

---

## 2. 生成的数据类型详解

### 数据结构

每个样本包含：

```python
{
    "features": np.array([f1, f2, f3, f4, f5, f6, f7]),  # 7 维特征
    "ds_expert": 0.0123,        # 专家建议的步长 (连续值)
    "y_restart": 0,             # 是否需要 SVD 重启 (0 或 1)
    
    # 元数据（用于分析）
    "matrix_type": "random_complex_dense",
    "matrix_size": 50,
    "matrix_id": "random_complex_dense_n50_0",
    "z": 0.523+0.123j,          # 当前复平面位置
    "u": np.array([...]),       # 左奇异向量 (n 维)
    "v": np.array([...]),       # 右奇异向量 (n 维)
    "residual": 1.23e-5,        # 残差 ||Mv - εu||
    "sigma_error": 2.34e-6,     # 奇异值误差 |σ - ε|
    "gamma": 0.876,             # 梯度 |u*v|
    "source": "expert" | "dagger",  # 数据来源
}
```

### 7 维特征详解

| 特征 | 符号 | 物理意义 | 计算方式 |
|------|------|----------|----------|
| f1 | `|u*Mv - ε|` | 奇异值偏差 | Rayleigh 商近似 |
| f2 | `|1 - ‖u‖|` | 左向量模长漂移 | 归一化误差 |
| f3 | `|1 - ‖v‖|` | 右向量模长漂移 | 归一化误差 |
| f4 | `‖Mv - εu‖` | 残差范数 | 直接计算 |
| f5 | `|u*v|` | 复梯度模长 | 内积 |
| f6 | `|Δarg(u*v)|` | 切向变化率 (曲率) | 角度差 |
| f7 | `N_iter` | 伪逆求解迭代次数 | Krylov 求解器 |

---

## 3. 数据生成流程

### 3.1 数据是什么？

**一条数据 = 伪谱等高线上的一个点 + 专家标注**

具体来说：
- **输入**: 7 维特征（与矩阵维度 n 无关）
- **标签 1**: 最优步长 `ds_expert` (回归任务)
- **标签 2**: 是否需要 SVD 重启 `y_restart` (分类任务)

### 3.2 生成流程图解

```
┌─────────────────────────────────────────────────────────────┐
│                    数据生成流水线                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: 生成矩阵 A                                          │
│  ┌──────────────┐                                          │
│  │ 矩阵类型      │──→ A (n×n)                              │
│  │ 矩阵大小 n    │                                          │
│  └──────────────┘                                          │
│                                                             │
│  Step 2: 找到伪谱等高线上的起点 z₀                            │
│  ┌──────────────┐                                          │
│  │ 射线搜索      │──→ z₀ (满足 σ_min(z₀I-A) = ε)           │
│  │ Brent 方法    │                                          │
│  └──────────────┘                                          │
│                                                             │
│  Step 3: 在 z₀ 处计算精确 SVD                                 │
│  ┌──────────────┐                                          │
│  │ 完整 SVD      │──→ (σ₀, u₀, v₀)                         │
│  │ (昂贵但精确)  │                                          │
│  └──────────────┘                                          │
│                                                             │
│  Step 4: 专家求解器生成轨迹                                   │
│  ┌──────────────┐                                          │
│  │ 自适应 RK45   │──→ 轨迹 {(z_t, u_t, v_t)}               │
│  │ (专家策略)    │                                          │
│  └──────────────┘                                          │
│                                                             │
│  Step 5: 提取特征 + 标注                                     │
│  ┌──────────────┐                                          │
│  │ 特征提取      │──→ 7 维特征                              │
│  │ 专家标注      │──→ ds_expert, y_restart                 │
│  └──────────────┘                                          │
│                                                             │
│  Step 6: DAgger 扰动增强                                     │
│  ┌──────────────┐                                          │
│  │ 状态扰动      │──→ 扰动状态                              │
│  │ 专家查询      │──→ 恢复动作标注                          │
│  └──────────────┘                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 详细步骤

#### Step 1: 生成矩阵

```python
# 例如：生成 50×50 随机复矩阵
A = np.random.randn(50, 50) + 1j * np.random.randn(50, 50)
```

#### Step 2: 找到等高线起点

使用**射线搜索 + Brent 二分法**：

```python
from scipy.optimize import brentq

def sigma_minus_epsilon(r):
    z = r * exp(i * angle)  # 沿射线方向
    M = z*I - A
    sigma_min = min(svd(M))  # 最小奇异值
    return sigma_min - epsilon  # 目标：sigma_min = epsilon

# 二分搜索找到 r
r_sol = brentq(sigma_minus_epsilon, 0.1, 5.0)
z0 = r_sol * exp(i * angle)
```

**验证**：确保 `σ_min(z₀I - A) ≈ ε`

#### Step 3: 计算精确 SVD

```python
M = z0 * I - A
U, S, Vh = np.linalg.svd(M, full_matrices=False)

# 找到最小奇异值对应的奇异向量
idx = argmin(S)
sigma_0 = S[idx]  # 应该 ≈ epsilon
u_0 = U[:, idx]
v_0 = Vh[idx, :].conj()
```

**这是"标注"的关键步骤**：使用**传统稠密 SVD**（`np.linalg.svd`），计算成本 O(n³)，但结果精确。

#### Step 4: 专家求解器生成轨迹

专家求解器是一个**高精度自适应 RK45 积分器**：

```python
from scipy.integrate import RK45

# 打包状态
y0 = pack(z0, u0, v0)

# 创建 RK45 求解器
rk_solver = RK45(
    fun=ode_rhs,  # ODE 右侧函数 (Eq. 13)
    t0=0.0,
    y0=y0,
    t_bound=1.0,
    max_step=0.1,      # 最大步长
    rtol=1e-8,         # 相对容差
    atol=1e-8,         # 绝对容差
    first_step=0.01,   # 初始步长
)

# 执行单步
rk_solver.step()

# 专家采用的步长 = RK45 自适应选择的步长
ds_expert = rk_solver.step_size

# 新状态
z_next, u_next, v_next = unpack(rk_solver.y)
```

**关键**：RK45 会根据局部截断误差**自动调整步长**：
- 平缓区域 → 大步长 (如 0.1)
- 陡峭区域 → 小步长 (如 0.001)

这个 `ds_expert` 就是**回归任务的 Ground Truth**。

#### Step 5: 决定是否重启（分类标签）

```python
# 计算当前残差
residual = ||M*v - epsilon*u||

# 如果残差过大，专家会选择 SVD 重启
if residual > drift_threshold:
    y_restart = 1  # 需要重启
    ds_expert = 0.0  # 重启时步长为 0
else:
    y_restart = 0  # 不需要重启
```

**分类任务的 Ground Truth**：`y_restart ∈ {0, 1}`

#### Step 6: 提取 7 维特征

```python
features = extract_features(z, u, v, A, epsilon)
# 返回 7 维向量
```

#### Step 7: DAgger 扰动增强

对轨迹上的每个点，添加扰动并重新查询专家：

```python
# 扰动状态
z_pert = z + noise
u_pert = u + noise
v_pert = v + noise

# 查询专家（在扰动状态下）
ds_recover, y_restart_recover = expert_query(z_pert, u_pert, v_pert)

# 添加到数据集
```

这教会网络如何从"糟糕状态"恢复。

---

## 4. 标注算法总结

### 标注流程

```
输入：矩阵 A, 伪谱水平 ε, 起点 z₀
        ↓
[1] 精确 SVD (np.linalg.svd)
    输出：(σ₀, u₀, v₀)
        ↓
[2] 自适应 RK45 积分一步
    输出：ds_expert (RK45 自动选择的步长)
        ↓
[3] 计算残差
    if ||Mv - εu|| > threshold:
        y_restart = 1
    else:
        y_restart = 0
        ↓
输出：(7 维特征，ds_expert, y_restart)
```

### 使用的算法

| 步骤 | 算法 | 复杂度 | 用途 |
|------|------|--------|------|
| 找起点 | Brent 二分法 | O(n³·log) | 找到等高线上的点 |
| 初始标注 | 稠密 SVD | O(n³) | 精确计算 (u₀, v₀) |
| 轨迹生成 | 自适应 RK45 | O(n²·k) | 生成专家轨迹 |
| 伪逆求解 | MINRES/GMRES | O(n²·iter) | ODE 积分中的线性系统 |
| 重启决策 | 残差检查 | O(n²) | 判断是否需要 SVD |

### 为什么需要传统 SVD？

1. **初始标注必须精确**：起点不精确会导致整个轨迹偏离
2. **重启时校正**：当 ODE 积分误差累积过大时，需要 SVD"重启"
3. **验证基准**：用于评估神经网络预测的准确性

**但 SVD 很昂贵** (O(n³))，所以我们训练神经网络来：
- 预测最优步长 `ds` (避免 RK45 的误差估计成本)
- 预测何时重启 `y_restart` (避免频繁调用 SVD)

---

## 5. 数据集示例

### 生成一个小型数据集

```bash
python scripts/generate_large_dataset.py \
    --target-samples 1000 \
    --matrix-sizes 20 30 \
    --matrix-types random_complex_dense \
    --trajectories-per-type 2 \
    --max-steps 50 \
    --dagger-factor 1 \
    --output-dir data/demo
```

### 查看数据

```python
import numpy as np

# 加载数据
data = np.load("data/demo/dataset_full.npz")

print("特征形状:", data["features"].shape)
# (1000, 7)

print("步长范围:", data["ds_expert"].min(), "-", data["ds_expert"].max())
# 0.0001 - 0.1

print("重启比例:", data["y_restart"].mean())
# 0.08 (8%)

# 查看一个样本
idx = 0
print("特征:", data["features"][idx])
# [0.0023, 0.0001, 0.0002, 0.0034, 0.876, 0.012, 0.156]
print("专家步长:", data["ds_expert"][idx])
# 0.0123
print("需要重启:", data["y_restart"][idx])
# 0
```

---

## 6. 使用 1024×1024 大矩阵

### 生成包含大矩阵的数据集

```bash
python scripts/generate_large_dataset.py \
    --target-samples 10000 \
    --matrix-sizes 100 256 512 1024 \
    --matrix-types random_complex_dense random_hermitian ill_conditioned \
    --trajectories-per-type 5 \
    --max-steps 100 \
    --dagger-factor 1 \
    --output-dir data/large_matrices \
    --seed 0
```

### 时间估算

| 矩阵大小 | 每步时间 | 100 步轨迹 | 5 条轨迹 |
|----------|----------|-----------|---------|
| 100×100 | ~20 秒 | 30 分钟 | 2.5 小时 |
| 256×256 | ~2 分钟 | 3 小时 | 15 小时 |
| 512×512 | ~10 分钟 | 16 小时 | 3.5 天 |
| 1024×1024 | ~1 小时 | 4 天 | 20 天 |

**建议**：
- 如果需要使用 1024×1024 矩阵，建议：
  1. 减少 `--target-samples` (如 1000)
  2. 减少 `--trajectories-per-type` (如 2-3)
  3. 减少 `--max-steps` (如 50)
  4. 使用稀疏矩阵加速

### 使用稀疏矩阵

```python
# 修改 generate_large_dataset.py
# 添加稀疏矩阵类型

@staticmethod
def random_sparse_large(n: int, density: float = 0.01, seed: int = 0):
    """稀疏矩阵 (适合大规模)"""
    from scipy import sparse
    rng = np.random.default_rng(seed)
    real_part = sparse.random(n, n, density=density, data_rng=rng)
    imag_part = sparse.random(n, n, density=density, data_rng=rng)
    return real_part + 1j * imag_part
```

**优势**：
- 内存：从 O(n²) 降到 O(n·density)
- 矩阵 - 向量乘法：从 O(n²) 降到 O(n·density)
- 适合 1024×1024 及以上规模

---

## 7. 数据可视化

### 绘制轨迹

```python
import matplotlib.pyplot as plt

data = np.load("data/demo/dataset_full.npz")

# 假设数据集中包含 z 的实部和虚部
# (需要修改 generate_large_dataset.py 保存这些信息)

plt.figure(figsize=(8, 8))
plt.plot(z_real, z_imag, 'b.-', markersize=2)
plt.xlabel('Re(z)')
plt.ylabel('Im(z)')
plt.title('Pseudospectrum Contour Trajectory')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.savefig('trajectory.png')
```

### 绘制特征分布

```python
feature_names = ['f1:σ误差', 'f2:u 漂移', 'f3:v 漂移', 
                 'f4:残差', 'f5:梯度模', 'f6:曲率', 'f7:迭代数']

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, name in enumerate(feature_names):
    axes[i].hist(data["features"][:, i], bins=50, alpha=0.7)
    axes[i].set_title(name)
    axes[i].grid(True, alpha=0.3)

axes[-1].axis('off')
plt.tight_layout()
plt.savefig('feature_distribution.png')
```

---

## 8. 总结

### 数据是什么？

**每个样本 = 伪谱等高线上的一个点 + 专家标注**

- **输入**: 7 维特征（与矩阵维度无关）
- **标签 1**: 最优步长 `ds_expert` (由 RK45 自适应选择)
- **标签 2**: 重启决策 `y_restart` (由残差阈值决定)

### 标注用什么算法？

1. **初始标注**: 传统稠密 SVD (`np.linalg.svd`) - O(n³)
2. **轨迹生成**: 自适应 RK45 积分器 - 自动选择步长
3. **重启决策**: 残差检查 - `||Mv - εu|| > threshold`

### 矩阵大小有限制吗？

**没有固定限制**，可以指定任意大小：
- 小型：16×16, 32×32 (快速测试)
- 中型：50×50, 100×100 (生产训练)
- 大型：256×256, 512×512 (大规模研究)
- 超大型：1024×1024 (需要稀疏矩阵或超级计算机)

### 数据多样性如何？

- **7 种矩阵类型**: 稠密、Hermitian、实矩阵、结构化、病态、正规...
- **多种维度**: 30, 50, 80, 100, 256, 512, 1024...
- **多个起点**: 每条轨迹从不同角度开始
- **DAgger 增强**: 2-3 倍扰动样本

---

去服务器执行时，根据你的需求选择合适的矩阵大小：

```bash
# 如果要用 1024×1024 大矩阵
python scripts/generate_large_dataset.py \
    --target-samples 1000 \
    --matrix-sizes 100 256 512 1024 \
    --trajectories-per-type 2 \
    --max-steps 50 \
    --output-dir data/large_scale_test
```
