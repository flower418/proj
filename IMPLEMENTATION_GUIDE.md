# 神经增强子空间追踪算法 - 代码实现指南

## 1. 项目结构规划

```
pseudospectrum_tracker/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   ├── default.yaml          # 默认配置
│   └── training.yaml         # 训练配置
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── manifold_ode.py   # 核心 ODE 系统 (Eq. 13)
│   │   ├── pseudoinverse.py  # 伪逆求解器
│   │   └── contour_tracker.py # 等高线追踪主循环
│   ├── nn/
│   │   ├── __init__.py
│   │   ├── controller.py     # 神经网络控制器
│   │   ├── features.py       # 状态特征提取
│   │   └── loss.py           # 损失函数
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── rk4.py            # Runge-Kutta 积分器
│   │   └── krylov.py         # Krylov 子空间迭代求解器
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── svd.py            # SVD 工具函数
│   │   ├── metrics.py        # 评估指标
│   │   └── visualization.py  # 可视化
│   └── train/
│       ├── __init__.py
│       ├── data_generator.py # 专家数据生成
│       └── trainer.py        # 训练循环
├── tests/
│   ├── test_manifold_ode.py
│   ├── test_controller.py
│   └── test_integration.py
└── scripts/
    ├── train_controller.py   # 训练脚本
    └── run_tracking.py       # 推理脚本
```

---

## 2. 核心模块实现细节

### 2.1 `src/core/manifold_ode.py` - 微分流形系统

**职责**：实现公式 (13) 的完整 ODE 系统

```python
# 关键函数签名
class ManifoldODE:
    def __init__(self, A: np.ndarray, epsilon: float):
        """
        :param A: n×n 目标矩阵
        :param epsilon: 伪谱等高线水平
        """
        
    def compute_dz_ds(self, z: complex, u: np.ndarray, v: np.ndarray) -> complex:
        """
        计算复平面上等高线的切向方向 (Eq. 8)
        dz/ds = i * (v* u) / |v* u|
        
        :return: 单位切向量 (|dz/ds| = 1)
        """
        
    def compute_dv_ds(self, z: complex, u: np.ndarray, v: np.ndarray, 
                      dz_ds: complex) -> np.ndarray:
        """
        计算右奇异向量的演化方程 (Eq. 12a)
        dv/ds = -(M*M - ε²I)⁺ · (dz/ds · M*v + ε · d(z̄)/ds · u)
        
        关键点:
        1. M = zI - A
        2. 使用伪逆 (·)⁺ 处理奇异性
        3. 结果自动垂直于 v (无需额外投影)
        """
        
    def compute_du_ds(self, z: complex, u: np.ndarray, v: np.ndarray,
                      dz_ds: complex) -> np.ndarray:
        """
        计算左奇异向量的演化方程 (Eq. 12b)
        du/ds = -(MM* - ε²I)⁺ · (d(z̄)/ds · Mu + ε · dz/ds · v)
        """
        
    def get_full_derivatives(self, z, u, v) -> Tuple[complex, np.ndarray, np.ndarray]:
        """
        一次性返回 (dz/ds, du/ds, dv/ds)
        用于 ODE 求解器的右侧函数评估
        """
```

**实现要点**：
- 缓存 `M = zI - A` 的计算，避免重复矩阵减法
- `M*v` 和 `M*u` 用矩阵 - 向量乘法，不要构造完整矩阵
- 伪逆求解是性能瓶颈，需单独优化 (见 2.2)

---

### 2.2 `src/core/pseudoinverse.py` - 摩尔 - 彭若斯伪逆求解器

**职责**：高效求解 `(H - σ²I)⁺ b` 形式的线性系统

```python
class PseudoinverseSolver:
    def __init__(self, method: str = 'minres', tol: float = 1e-8, 
                 max_iter: int = 1000):
        """
        :param method: 'minres' | 'cg' | 'svd' (小规模用)
        :param tol: Krylov 求解器容差
        :param max_iter: 最大迭代次数
        """
        
    def solve(self, H: callable, sigma_sq: float, b: np.ndarray, 
              null_vector: np.ndarray = None) -> np.ndarray:
        """
        求解 (H - σ²I)⁺ b
        
        :param H: 线性算子 (支持 H @ x 形式调用)
        :param sigma_sq: σ² 值
        :param b: 右侧向量 (必须垂直于零空间)
        :param null_vector: 已知的零空间向量 (用于投影验证)
        :return: 最小范数解 x
        
        关键步骤:
        1. 验证 b ⟂ null_vector (正交性检查)
        2. 使用 Krylov 方法迭代求解
        3. 可选：投影掉零空间分量 x ← x - (v*x)v
        """
        
    def get_iteration_count(self) -> int:
        """返回上次求解的迭代次数 (用于特征 f_7)"""
```

**实现要点**：
- 使用 `scipy.sparse.linalg.minres` 或 `cg`
- 实现矩阵-free 的线性算子 (LinearOperator)
- 对于小规模问题 (n < 500)，可直接用稠密 SVD

---

### 2.3 `src/core/contour_tracker.py` - 等高线追踪主循环

**职责**：实现算法主循环 (伪代码第 2 部分)

```python
class ContourTracker:
    def __init__(self, A: np.ndarray, epsilon: float, 
                 ode_system: ManifoldODE,
                 controller: Optional[NNController] = None,
                 svd_solver: callable = None):
        """
        :param A: 目标矩阵
        :param epsilon: 伪谱水平
        :param ode_system: ODE 系统实例
        :param controller: 神经网络控制器 (None = 纯 ODE 基线)
        :param svd_solver: 精确 SVD 计算函数
        """
        
    def initialize(self, z0: complex) -> Tuple[np.ndarray, np.ndarray]:
        """
        在起点 z0 处执行初始 SVD
        返回精确的 (u0, v0)
        """
        
    def track(self, z0: complex, max_steps: int = 1000) -> Dict:
        """
        主追踪循环
        
        :param z0: 等高线起点
        :param max_steps: 最大步数
        :return: 字典包含：
            - 'trajectory': z 的复数轨迹
            - 'u_history': u 向量历史
            - 'v_history': v 向量历史
            - 'restart_indices': 执行 SVD 重启的步索引
            - 'step_sizes': 每步的实际步长
        """
        
    def extract_state_features(self, z, u, v, prev_state=None) -> np.ndarray:
        """
        提取 7 维状态特征向量 (见第 3 节)
        """
        
    def check_closure(self, z_current: complex, z_start: complex, 
                      min_steps: int = 10) -> bool:
        """
        检测等高线是否闭合
        条件：|z_current - z_start| < tol 且已走至少 min_steps 步
        """
```

**主循环伪代码实现**：
```python
def track(self, z0, max_steps=1000):
    # 1. 初始化 (昂贵 SVD)
    z, u, v = self.initialize(z0)
    trajectory = [z]
    
    for step in range(max_steps):
        # 2. 特征提取
        state = self.extract_state_features(z, u, v)
        
        # 3. 神经网络决策
        if self.controller is not None:
            ds, need_restart = self.controller.predict(state)
        else:
            ds = self.fixed_step_size  # 基线：固定步长
            need_restart = False
        
        # 4. 分支执行
        if need_restart:
            z, u, v = self.exact_svd_restart(z)
            record_restart(step)
        else:
            # ODE 积分步
            dz, du, dv = self.ode_system.get_full_derivatives(z, u, v)
            z = z + ds * dz
            u = u + ds * du
            v = v + ds * dv
            # 归一化 (低成本)
            u /= np.linalg.norm(u)
            v /= np.linalg.norm(v)
        
        trajectory.append(z)
        
        # 5. 闭合检测
        if self.check_closure(z, z0) and step > 10:
            break
    
    return compile_results(trajectory)
```

---

### 2.4 `src/nn/controller.py` - 神经网络控制器

**职责**：两头部 MLP，输出步长和重启决策

```python
class NNController(nn.Module):
    def __init__(self, input_dim: int = 7, hidden_dims: List[int] = [64, 64],
                 dropout: float = 0.1):
        super().__init__()
        
        # 共享特征提取层
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(h_dim)
            ])
            prev_dim = h_dim
        self.shared_encoder = nn.Sequential(*layers)
        
        # 步长回归头部 (Softplus 确保正数)
        self.step_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Softplus()  # 输出 > 0
        )
        
        # 重启决策分类头部 (Sigmoid 输出概率)
        self.restart_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()  # 输出 ∈ (0, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: 形状 [batch, 7] 的状态特征
        :return: (step_size [batch, 1], restart_prob [batch, 1])
        """
        features = self.shared_encoder(x)
        ds = self.step_head(features)
        p_restart = self.restart_head(features)
        return ds, p_restart
    
    def predict(self, state_np: np.ndarray) -> Tuple[float, bool]:
        """
        单步推理接口 (NumPy → NumPy)
        :param state_np: 7 维状态向量
        :return: (ds, need_restart)
        """
```

---

### 2.5 `src/nn/features.py` - 状态特征提取

**职责**：实现 7 维状态特征向量

```python
def extract_features(z: complex, u: np.ndarray, v: np.ndarray,
                     A: np.ndarray, epsilon: float,
                     prev_gamma_arg: float = None,
                     prev_solver_iters: int = 0) -> np.ndarray:
    """
    提取 7 维标量特征向量 (与矩阵维度 n 无关!)
    
    返回 np.array([f1, f2, f3, f4, f5, f6, f7])
    """
    M = z * np.eye(len(u)) - A  # 或 sparse 矩阵
    
    # === 数值漂移群 (Drift Indicators) ===
    # f1: 奇异值偏差 (Rayleigh 商近似)
    sigma_approx = np.real(u.conj() @ M @ v)
    f1 = np.abs(sigma_approx - epsilon)
    
    # f2, f3: 模长漂移
    f2 = np.abs(1.0 - np.linalg.norm(u))
    f3 = np.abs(1.0 - np.linalg.norm(v))
    
    # f4: 残差范数
    residual = M @ v - epsilon * u
    f4 = np.linalg.norm(residual)
    
    # === 局部几何群 (Geometric Topography) ===
    # f5: 复梯度模长
    gamma = u.conj() @ v
    f5 = np.abs(gamma)
    
    # f6: 切向方向变化率 (曲率估计)
    if prev_gamma_arg is not None:
        f6 = np.abs(np.angle(gamma) - prev_gamma_arg)
        # 包裹到 [-π, π]
        f6 = np.minimum(f6, 2*np.pi - f6)
    else:
        f6 = 0.0
    
    # === 算力健康度 (Solver Health) ===
    # f7: 上次伪逆求解迭代次数
    f7 = prev_solver_iters
    
    return np.array([f1, f2, f3, f4, f5, f6, f7], dtype=np.float32)
```

---

### 2.6 `src/nn/loss.py` - 损失函数

**职责**：实现 Log-MSE + 加权 BCE

```python
class ControllerLoss(nn.Module):
    def __init__(self, lambda_step: float = 1.0, lambda_restart: float = 1.0,
                 alpha_restart: float = 0.9):
        """
        :param lambda_step: 步长损失权重
        :param lambda_restart: 重启损失权重
        :param alpha_restart: 重启类别的权重 (处理不平衡)
        """
        super().__init__()
        self.lambda_step = lambda_step
        self.lambda_restart = lambda_restart
        self.alpha = alpha_restart
        
    def forward(self, ds_pred, ds_expert, p_restart, y_restart):
        """
        :param ds_pred: 预测步长 [batch]
        :param ds_expert: 专家步长 [batch] (Ground Truth)
        :param p_restart: 预测重启概率 [batch]
        :param y_restart: 专家重启标签 [batch] ∈ {0, 1}
        
        :return: total_loss, step_loss, restart_loss
        """
        # Log-MSE for step size
        log_ds_pred = torch.log(ds_pred + 1e-8)
        log_ds_expert = torch.log(ds_expert + 1e-8)
        step_loss = torch.mean((log_ds_pred - log_ds_expert) ** 2)
        
        # Weighted BCE for restart
        eps = 1e-8
        bce_loss = -(
            self.alpha * y_restart * torch.log(p_restart + eps) +
            (1 - self.alpha) * (1 - y_restart) * torch.log(1 - p_restart + eps)
        )
        restart_loss = torch.mean(bce_loss)
        
        total_loss = self.lambda_step * step_loss + self.lambda_restart * restart_loss
        return total_loss, step_loss, restart_loss
```

---

### 2.7 `src/train/data_generator.py` - 专家数据生成

**职责**：使用高精度自适应求解器生成训练数据

```python
class ExpertDataGenerator:
    def __init__(self, A: np.ndarray, epsilon: float,
                 expert_tol: float = 1e-8,
                 drift_threshold: float = 1e-4):
        """
        :param expert_tol: 专家求解器容差 (RK45 自适应)
        :param drift_threshold: 触发 SVD 的残差阈值
        """
        
    def generate_trajectory(self, z0: complex, max_steps: int = 500) -> List[Dict]:
        """
        运行专家求解器，收集完整轨迹数据
        
        返回 List[Dict], 每个 Dict 包含:
        {
            'z': complex,
            'u': np.ndarray,
            'v': np.ndarray,
            'ds_expert': float,  # 专家采用的安全步长
            'y_restart': int,    # 是否触发 SVD
            'features': np.ndarray  # 7 维状态
        }
        """
        
    def add_state_perturbations(self, trajectory: List[Dict], 
                                noise_std: float = 0.01) -> List[Dict]:
        """
        DAgger 策略：对状态注入高斯扰动
        模拟数值漂移，让网络学会"恢复"行为
        """
```

**专家求解器实现要点**：
- 使用 `scipy.integrate.solve_ivp` with `method='RK45'`
- 设置 `rtol=1e-8, atol=1e-8`
- 监控残差 `f4`，超过阈值时标记 `y_restart=1` 并执行 SVD
- 记录求解器自动选择的最大安全步长作为 `ds_expert`

---

### 2.8 `src/train/trainer.py` - 训练循环

```python
class ControllerTrainer:
    def __init__(self, model: NNController, loss_fn: ControllerLoss,
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cuda'):
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """单 epoch 训练"""
        
    def train(self, train_dataset, val_dataset, epochs: int, 
              early_stop_patience: int = 10):
        """完整训练流程"""
```

---

## 3. 状态特征详细说明

| 特征 | 符号 | 物理意义 | 计算成本 | 典型范围 |
|------|------|----------|----------|----------|
| f1 | $\|u^* M v\| - \epsilon$ | 奇异值偏差 | O(n) | 1e-6 ~ 1e-1 |
| f2 | $|1 - \|u\||$ | 左向量模长漂移 | O(n) | 1e-8 ~ 1e-2 |
| f3 | $|1 - \|v\||$ | 右向量模长漂移 | O(n) | 1e-8 ~ 1e-2 |
| f4 | $\|Mv - \epsilon u\|$ | 残差置信度 | O(n²) | 1e-6 ~ 1 |
| f5 | $|u^* v|$ | 梯度模长 (曲率指示) | O(n) | 1e-3 ~ 1 |
| f6 | $|\Delta \arg(u^* v)|$ | 切向变化率 | O(1) | 0 ~ π/4 |
| f7 | $N_{iter}$ | 伪逆求解迭代次数 | O(1) | 5 ~ 200 |

**归一化建议**：
- f1, f2, f3, f4：取对数后归一化到 [-1, 1]
- f5：直接用原值 (已在 [0, 1])
- f6：除以 π 归一化
- f7：除以 max_iter 归一化

---

## 4. 关键算法参数配置

```yaml
# configs/default.yaml
ode:
  epsilon: 0.1              # 伪谱水平
  initial_step_size: 0.01   # 初始步长 ds
  min_step_size: 1e-6       # 最小允许步长
  max_step_size: 0.1        # 最大允许步长
  
solver:
  method: minres            # Krylov 方法
  tol: 1e-8                 # 伪逆求解容差
  max_iter: 500             # 最大迭代次数
  
tracker:
  max_steps: 1000           # 最大追踪步数
  closure_tol: 1e-3         # 闭合检测容差
  restart_drift_threshold: 1e-4  # 触发重启的漂移阈值
  
controller:
  hidden_dims: [64, 64]
  dropout: 0.1
  step_size_min: 1e-4
  step_size_max: 0.1
  
training:
  batch_size: 128
  learning_rate: 1e-3
  epochs: 100
  lambda_step: 1.0
  lambda_restart: 5.0       # 重启损失权重更高
  alpha_restart: 0.9        # 正样本权重
  noise_std: 0.01           # DAgger 扰动强度
```

---

## 5. 测试策略

### 5.1 单元测试

```python
# tests/test_manifold_ode.py
def test_dz_ds_unit_norm():
    """验证 dz/ds 的模长恒为 1"""
    
def test_dv_orthogonal_to_v():
    """验证 dv/ds ⟂ v (数值误差 < 1e-10)"""
    
def test_conservation_sigma():
    """验证沿轨线 σ(z) ≈ epsilon (漂移 < tol)"""

# tests/test_controller.py
def test_controller_output_positive():
    """验证步长输出严格为正"""
    
def test_controller_restart_probability_range():
    """验证重启概率 ∈ (0, 1)"""
```

### 5.2 集成测试

```python
# tests/test_integration.py
def test_closed_contour():
    """测试完整闭合等高线追踪"""
    # 使用小规模随机矩阵，验证轨迹闭合
    
def test_neural_vs_baseline():
    """对比神经网络 vs 固定步长基线的性能"""
    # 指标：SVD 调用次数、总运行时间、轨迹精度
```

---

## 6. 性能优化建议

1. **矩阵 - 向量乘法**：
   - 对于大规模稀疏矩阵，使用 `scipy.sparse`
   - 避免构造 `M*M` 等稠密矩阵，用 `A.T @ (A @ x)` 代替

2. **伪逆求解缓存**：
   - 相邻步的 `z` 变化小，可用上一步的解作为 Krylov 初值

3. **批量化特征提取**：
   - 训练数据生成时，用 NumPy 向量化操作替代循环

4. **GPU 加速**：
   - 神经网络推理用 CUDA
   - 大规模矩阵运算可用 CuPy 替代 NumPy

---

## 7. 预期输出与可视化

```python
# scripts/visualize_results.py
def plot_pseudospectrum_contour(trajectory, A, epsilon):
    """绘制伪谱等高线"""
    
def plot_controller_decisions(step_sizes, restart_points):
    """可视化控制器的步长选择和重启决策"""
    
def plot_feature_evolution(features_history):
    """绘制 7 维特征随时间的演化"""
```

---

## 8. 开发优先级

**Phase 1 (核心功能)**：
1. `manifold_ode.py` - ODE 系统
2. `pseudoinverse.py` - 伪逆求解器
3. `contour_tracker.py` - 主追踪循环
4. 基础测试

**Phase 2 (基线验证)**：
1. 固定步长 ODE 追踪
2. 与纯 SVD 方法对比精度/速度

**Phase 3 (神经网络)**：
1. `controller.py` + `features.py`
2. `data_generator.py` - 专家数据
3. `loss.py` + `trainer.py`
4. 训练和评估

**Phase 4 (优化与完善)**：
1. 性能分析 (profiling)
2. 大规模矩阵测试
3. 文档和示例
