# 训练数据生成详细指南

## 1. 概述

### 1.1 为什么需要特殊的数据生成策略？

本项目的核心挑战：**训练数据无法通过解析解获得**。

- **专家标签**：最优步长 $ds^*$ 和重启决策 $y^*$ 来自"理想求解器"的行为
- **分布偏移问题**：仅在完美轨迹上训练 → 推理时遇到漂移状态会失效
- **解决方案**：DAgger (Dataset Aggregation) + 专家查询

### 1.2 数据生成流程总览

```
┌─────────────────────────────────────────────────────────────┐
│                    训练数据生成流水线                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: 专家轨迹生成                                        │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐│
│  │ 随机矩阵 A    │────▶│ 高精度 RK45   │────▶│ 完美轨迹 D₀   ││
│  │ + 起点 z₀    │     │ 求解器       │     │ (无扰动)     ││
│  └──────────────┘     └──────────────┘     └──────────────┘│
│                              │                                │
│                              ▼                                │
│  Step 2: DAgger 扰动增强                                         │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐│
│  │ 轨迹 D₀       │────▶│ 高斯噪声注入  │────▶│ 扰动状态 D'    ││
│  │              │     │              │     │              ││
│  └──────────────┘     └──────────────┘     └──────────────┘│
│                              │                                │
│                              ▼                                │
│  Step 3: 专家恢复查询                                           │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐│
│  │ 扰动状态 D'   │────▶│ 专家求解器    │────▶│ 恢复动作标签  ││
│  │              │     │ "如果这样..." │     │ (步长 + 重启) ││
│  └──────────────┘     └──────────────┘     └──────────────┘│
│                              │                                │
│                              ▼                                │
│  Step 4: 数据聚合                                                │
│  ┌──────────────┐     ┌──────────────┐                       │
│  │ D₀ ∪ D'      │────▶│ 最终训练集    │                       │
│  │              │     │              │                       │
│  └──────────────┘     └──────────────┘                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 专家求解器设计

### 2.1 专家求解器的核心职责

专家求解器是一个**设置了极严苛容差**的自适应 Runge-Kutta 求解器，它的作用是：

1. **提供 Ground Truth 步长**：专家在当前状态下能安全采用的最大步长
2. **提供 Ground Truth 重启标签**：何时必须调用 SVD 校正

### 2.2 专家求解器实现

```python
# src/train/expert_solver.py

import numpy as np
from scipy.integrate import solve_ivp, RK45
from typing import Tuple, List, Dict, Optional

class ExpertSolver:
    """
    高精度自适应 ODE 求解器 - 作为训练数据的"专家神谕"
    """
    
    def __init__(self, A: np.ndarray, epsilon: float,
                 rtol: float = 1e-8, atol: float = 1e-8,
                 max_step: float = 0.1,
                 drift_threshold: float = 1e-4,
                 svd_solver: callable = None):
        """
        :param A: n×n 目标矩阵
        :param epsilon: 伪谱水平
        :param rtol, atol: RK45 容差 (设置极严苛)
        :param max_step: 允许的最大步长
        :param drift_threshold: 触发 SVD 重启的残差阈值
        :param svd_solver: 精确 SVD 计算函数
        """
        self.A = A
        self.epsilon = epsilon
        self.n = A.shape[0]
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step
        self.drift_threshold = drift_threshold
        self.svd_solver = svd_solver or self._default_svd
        
        # 预计算
        self.I = np.eye(self.n)
        
    def _default_svd(self, z: complex) -> Tuple[np.ndarray, np.ndarray]:
        """默认 SVD 实现 (小规模用稠密，大规模用稀疏)"""
        M = z * self.I - self.A
        if self.n < 500:
            # 稠密 SVD
            U, S, Vh = np.linalg.svd(M, full_matrices=False)
            idx = np.argmin(S)
            sigma = S[idx]
            u = U[:, idx]
            v = Vh[idx, :].conj().T
            return u, v
        else:
            # 稀疏 SVD (用 scipy.sparse.linalg.svds)
            from scipy.sparse.linalg import svds
            # 实现略...
            pass
    
    def _ode_rhs(self, s: float, y: np.ndarray) -> np.ndarray:
        """
        ODE 右侧函数 dy/ds = f(y)
        
        状态向量 y = [Re(z), Im(z), Re(u₁), Im(u₁), ..., Re(vₙ), Im(vₙ)]
        总维度：2 + 2n + 2n = 4n + 2
        
        使用 Eq. 13 计算导数
        """
        # 解包状态
        z = y[0] + 1j * y[1]
        u = y[2:2+self.n] + 1j * y[2+self.n:2+2*self.n]
        v = y[2+2*self.n:2+3*self.n] + 1j * y[2+3*self.n:]
        
        # 计算 M = zI - A
        M = z * self.I - self.A
        M_H = M.conj().T
        
        # 计算复梯度 γ = u* v
        gamma = np.vdot(u, v)  # u^H v
        gamma_norm = np.abs(gamma)
        
        # Eq. 8: dz/ds = i * γ̄ / |γ|
        if gamma_norm < 1e-12:
            # 梯度接近 0，危险区域
            dz_ds = 1j * np.exp(-1j * np.angle(gamma + 1e-12))
        else:
            dz_ds = 1j * np.conj(gamma) / gamma_norm
        
        # 计算伪逆项 (M*M - ε²I)⁺ 和 (MM* - ε²I)⁺
        # 使用 Krylov 迭代求解线性系统
        sigma_sq = self.epsilon ** 2
        
        # 右侧向量
        rhs_v = dz_ds * (M_H @ v) + self.epsilon * np.conj(dz_ds) * u
        rhs_u = np.conj(dz_ds) * (M @ u) + self.epsilon * dz_ds * v
        
        # 求解 (M*M - σ²I) dv/ds = -rhs_v
        # 使用 MINRES (因为 M*M - σ²I 是对称的，可能不定)
        from scipy.sparse.linalg import minres, LinearOperator
        
        def matvec_V(x):
            return M_H @ (M @ x) - sigma_sq * x
        V_op = LinearOperator((self.n, self.n), matvec=matvec_V)
        
        dv_ds, info_v = minres(V_op, -rhs_v, tol=1e-10, maxiter=1000)
        
        def matvec_U(x):
            return M @ (M_H @ x) - sigma_sq * x
        U_op = LinearOperator((self.n, self.n), matvec=matvec_U)
        
        du_ds, info_u = minres(U_op, -rhs_u, tol=1e-10, maxiter=1000)
        
        # 打包导数
        dy_ds = np.zeros_like(y, dtype=np.float64)
        dy_ds[0] = np.real(dz_ds)
        dy_ds[1] = np.imag(dz_ds)
        dy_ds[2:2+self.n] = np.real(du_ds)
        dy_ds[2+self.n:2+2*self.n] = np.imag(du_ds)
        dy_ds[2+2*self.n:2+3*self.n] = np.real(dv_ds)
        dy_ds[2+3*self.n:] = np.imag(dv_ds)
        
        return dy_ds
    
    def compute_residual(self, z: complex, u: np.ndarray, 
                         v: np.ndarray) -> float:
        """
        计算残差 ||Mv - εu|| (特征 f4)
        """
        M = z * self.I - self.A
        residual = np.linalg.norm(M @ v - self.epsilon * u)
        return residual
    
    def compute_sigma_approx(self, z: complex, u: np.ndarray, 
                             v: np.ndarray) -> float:
        """
        通过 Rayleigh 商近似奇异值 σ ≈ |u* M v|
        """
        M = z * self.I - self.A
        sigma_approx = np.abs(np.vdot(u, M @ v))
        return sigma_approx
    
    def generate_expert_trajectory(self, z0: complex, 
                                    max_steps: int = 500,
                                    min_steps_before_restart: int = 5
                                   ) -> List[Dict]:
        """
        生成专家轨迹 (无扰动)
        
        :param z0: 起点
        :param max_steps: 最大步数
        :param min_steps_before_restart: 两次 SVD 之间的最小步数
        
        :return: 轨迹数据列表
        """
        # === 步骤 1: 初始 SVD ===
        u, v = self.svd_solver(z0)
        z = z0
        
        # 归一化
        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)
        
        trajectory = []
        steps_since_restart = 0
        last_restart_z = z0
        
        # === 步骤 2: 主追踪循环 ===
        for step in range(max_steps):
            # 记录当前状态 (用于特征提取)
            state_before = {
                'z': z.copy(),
                'u': u.copy(),
                'v': v.copy(),
                'step': step
            }
            
            # 计算当前残差和奇异值偏差
            residual = self.compute_residual(z, u, v)
            sigma_approx = self.compute_sigma_approx(z, u, v)
            sigma_error = np.abs(sigma_approx - self.epsilon)
            
            # === 决策 1: 是否需要 SVD 重启？===
            need_restart = False
            
            # 条件 A: 残差过大
            if residual > self.drift_threshold and steps_since_restart >= min_steps_before_restart:
                need_restart = True
                restart_reason = 'drift'
            
            # 条件 B: 奇异值偏差过大
            elif sigma_error > self.drift_threshold and steps_since_restart >= min_steps_before_restart:
                need_restart = True
                restart_reason = 'sigma_error'
            
            # 条件 C: 梯度接近 0 (曲率奇异点)
            gamma = np.abs(np.vdot(u, v))
            if gamma < 1e-6:
                need_restart = True
                restart_reason = 'singular_gamma'
            
            if need_restart:
                # 执行 SVD 重启
                u, v = self.svd_solver(z)
                u = u / np.linalg.norm(u)
                v = v / np.linalg.norm(v)
                steps_since_restart = 0
                last_restart_z = z
                
                # 记录重启事件
                trajectory.append({
                    **state_before,
                    'ds_expert': 0.0,  # 重启步长为 0
                    'y_restart': 1,
                    'restart_reason': restart_reason,
                    'residual': residual,
                    'sigma_error': sigma_error,
                    'gamma': gamma
                })
                continue
            
            # === 决策 2: 确定安全步长 ===
            # 使用自适应 RK45 单步，让求解器选择最大安全步长
            
            # 打包初始状态
            y0 = self._pack_state(z, u, v)
            
            # 创建 RK45 求解器 (单步模式)
            rk_solver = RK45(
                fun=self._ode_rhs,
                t0=0.0,
                y0=y0,
                t_bound=1.0,  # 虚拟终点
                max_step=self.max_step,
                rtol=self.rtol,
                atol=self.atol,
                first_step=0.01  # 初始尝试步长
            )
            
            # 执行单步
            rk_solver.step()
            
            # 专家采用的步长 = RK45 自适应选择的步长
            ds_expert = rk_solver.step_size
            
            # 获取新状态
            y_new = rk_solver.y
            z_new = y_new[0] + 1j * y_new[1]
            u_new = y_new[2:2+self.n] + 1j * y_new[2+self.n:2+2*self.n]
            v_new = y_new[2+2*self.n:2+3*self.n] + 1j * y_new[2+3*self.n:]
            
            # 归一化 (专家也做归一化)
            u_new = u_new / np.linalg.norm(u_new)
            v_new = v_new / np.linalg.norm(v_new)
            
            # === 记录数据点 ===
            trajectory.append({
                **state_before,
                'z_next': z_new,
                'u_next': u_new,
                'v_next': v_new,
                'ds_expert': ds_expert,
                'y_restart': 0,
                'residual': residual,
                'sigma_error': sigma_error,
                'gamma': gamma,
                'rk_info': rk_solver.step_size
            })
            
            # 更新状态
            z, u, v = z_new, u_new, v_new
            steps_since_restart += 1
            
            # === 闭合检测 ===
            if step > 10 and np.abs(z - z0) < 1e-3:
                print(f"等高线闭合于 step={step}")
                break
        
        return trajectory
    
    def _pack_state(self, z: complex, u: np.ndarray, 
                    v: np.ndarray) -> np.ndarray:
        """打包状态为实数向量"""
        y = np.zeros(2 + 4 * self.n, dtype=np.float64)
        y[0] = np.real(z)
        y[1] = np.imag(z)
        y[2:2+self.n] = np.real(u)
        y[2+self.n:2+2*self.n] = np.imag(u)
        y[2+2*self.n:2+3*self.n] = np.real(v)
        y[2+3*self.n:] = np.imag(v)
        return y
```

---

## 3. DAgger 扰动增强策略

### 3.1 为什么需要 DAgger？

**问题**：行为克隆 (Behavioral Cloning) 的复合误差问题

- 训练时：网络看到的都是"完美状态" (专家轨迹上的点)
- 推理时：网络看到的是"带噪声状态" (ODE 数值漂移累积)
- 结果：分布偏移 → 网络做出错误决策 → 误差进一步累积 → 崩溃

**DAgger 解决方案**：
1. 对专家轨迹注入扰动，模拟数值漂移
2. 查询专家："如果你处于这个糟糕状态，你会怎么做？"
3. 将扰动状态 + 专家恢复动作加入训练集

### 3.2 DAgger 实现

```python
# src/train/dagger_augmentation.py

import numpy as np
from typing import List, Dict, Tuple

class DAggerAugmenter:
    """
    DAgger (Dataset Aggregation) 扰动增强
    """
    
    def __init__(self, expert_solver: ExpertSolver,
                 feature_extractor: callable,
                 noise_scales: Dict[str, float] = None):
        """
        :param expert_solver: 专家求解器实例
        :param feature_extractor: 状态特征提取函数
        :param noise_scales: 各状态分量的噪声标准差
        """
        self.expert = expert_solver
        self.feature_extractor = feature_extractor
        
        # 默认噪声强度 (可根据矩阵规模调整)
        self.noise_scales = noise_scales or {
            'z_real': 1e-3,      # z 实部噪声
            'z_imag': 1e-3,      # z 虚部噪声
            'u': 1e-2,           # u 向量噪声
            'v': 1e-2,           # v 向量噪声
            'phase': 0.1         # 相位扰动
        }
        
    def perturb_state(self, z: complex, u: np.ndarray, v: np.ndarray,
                      seed: int = None) -> Tuple[complex, np.ndarray, np.ndarray]:
        """
        对状态注入高斯扰动
        
        :param z: 原始复平面位置
        :param u: 原始左奇异向量
        :param v: 原始右奇异向量
        :return: (z_perturbed, u_perturbed, v_perturbed)
        """
        if seed is not None:
            np.random.seed(seed)
        
        n = len(u)
        
        # === 扰动 z ===
        z_perturbed = z + (np.random.randn() * self.noise_scales['z_real'] +
                          1j * np.random.randn() * self.noise_scales['z_imag'])
        
        # === 扰动 u (保持近似单位长度) ===
        # 方法 1: 加性噪声 + 归一化
        u_perturbed = u + np.random.randn(n) * self.noise_scales['u'] + \
                     1j * np.random.randn(n) * self.noise_scales['u']
        u_perturbed = u_perturbed / np.linalg.norm(u_perturbed)
        
        # === 扰动 v (保持近似单位长度) ===
        v_perturbed = v + np.random.randn(n) * self.noise_scales['v'] + \
                     1j * np.random.randn(n) * self.noise_scales['v']
        v_perturbed = v_perturbed / np.linalg.norm(v_perturbed)
        
        return z_perturbed, u_perturbed, v_perturbed
    
    def query_expert_at_perturbed_state(self, z_pert: complex, 
                                         u_pert: np.ndarray, 
                                         v_pert: np.ndarray,
                                         original_ds: float) -> Tuple[float, int]:
        """
        在扰动状态查询专家，获取恢复动作
        
        策略：
        1. 计算扰动状态的残差
        2. 如果残差已经很大 → 专家会要求立即 SVD 重启
        3. 如果残差尚可 → 专家会选择一个保守的小步长
        
        :return: (ds_expert_recover, y_restart_expert)
        """
        # 计算扰动状态的残差
        residual_pert = self.expert.compute_residual(z_pert, u_pert, v_pert)
        sigma_error_pert = np.abs(
            self.expert.compute_sigma_approx(z_pert, u_pert, v_pert) - 
            self.expert.epsilon
        )
        
        # === 决策 1: 是否需要立即重启？===
        # 如果扰动导致残差超过阈值，专家会选择重启
        if residual_pert > self.expert.drift_threshold * 0.5:  # 更保守的阈值
            return 0.0, 1  # 重启，步长为 0
        
        if sigma_error_pert > self.expert.drift_threshold * 0.5:
            return 0.0, 1
        
        # === 决策 2: 选择保守步长 ===
        # 扰动越大，步长越保守
        perturbation_magnitude = (
            self.expert.compute_residual(z_pert, u_pert, v_pert) / 
            self.expert.drift_threshold
        )
        
        # 步长缩减因子 (扰动越大，步长越小)
        reduction_factor = np.clip(1.0 - perturbation_magnitude, 0.1, 1.0)
        ds_recover = original_ds * reduction_factor * 0.5  # 额外保守
        
        return ds_recover, 0
    
    def augment_trajectory(self, trajectory: List[Dict],
                           num_perturbations_per_point: int = 3
                          ) -> List[Dict]:
        """
        对整条轨迹进行 DAgger 增强
        
        :param trajectory: 原始专家轨迹
        :param num_perturbations_per_point: 每个点生成的扰动样本数
        :return: 增强后的数据集
        """
        augmented_data = []
        
        for i, point in enumerate(trajectory):
            z = point['z']
            u = point['u']
            v = point['v']
            ds_original = point['ds_expert']
            y_restart_original = point['y_restart']
            
            # 原始数据点 (保留)
            features = self.feature_extractor(z, u, v)
            augmented_data.append({
                'features': features,
                'ds_expert': ds_original,
                'y_restart': y_restart_original,
                'source': 'original',
                'trajectory_idx': i
            })
            
            # 跳过重启点 (重启点不需要扰动)
            if y_restart_original == 1:
                continue
            
            # 生成扰动样本
            for j in range(num_perturbations_per_point):
                z_pert, u_pert, v_pert = self.perturb_state(
                    z, u, v, seed=i * 100 + j
                )
                
                # 查询专家
                ds_recover, y_restart_recover = \
                    self.query_expert_at_perturbed_state(
                        z_pert, u_pert, v_pert, ds_original
                    )
                
                # 提取扰动状态的特征
                features_pert = self.feature_extractor(z_pert, u_pert, v_pert)
                
                augmented_data.append({
                    'features': features_pert,
                    'ds_expert': ds_recover,
                    'y_restart': y_restart_recover,
                    'source': 'dagger',
                    'trajectory_idx': i,
                    'perturbation_idx': j,
                    'residual_pert': self.expert.compute_residual(
                        z_pert, u_pert, v_pert
                    )
                })
        
        return augmented_data
```

---

## 4. 完整数据生成流水线

### 4.1 主数据生成器

```python
# src/train/data_generator.py

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path

class PseudospectrumDataset(Dataset):
    """
    PyTorch Dataset 封装
    """
    
    def __init__(self, data_list: List[Dict], 
                 normalize: bool = True,
                 log_transform_step: bool = True):
        """
        :param data_list: 数据列表 (来自 DAgger 增强)
        :param normalize: 是否归一化特征
        :param log_transform_step: 是否对步长取对数
        """
        self.data = data_list
        self.normalize = normalize
        self.log_transform_step = log_transform_step
        
        # 计算归一化统计量
        if normalize:
            self._compute_normalization_stats()
    
    def _compute_normalization_stats(self):
        """计算特征的均值和标准差"""
        features_stack = np.stack([d['features'] for d in self.data])
        self.feature_mean = np.mean(features_stack, axis=0)
        self.feature_std = np.std(features_stack, axis=0) + 1e-8
        
        # 步长统计量 (用于回归目标归一化)
        ds_values = np.array([d['ds_expert'] for d in self.data if d['ds_expert'] > 0])
        self.ds_mean = np.mean(np.log(ds_values + 1e-8))
        self.ds_std = np.std(np.log(ds_values + 1e-8)) + 1e-8
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        point = self.data[idx]
        
        # 特征
        features = point['features'].copy()
        if self.normalize:
            features = (features - self.feature_mean) / self.feature_std
        
        # 步长目标 (可选对数变换)
        ds = point['ds_expert']
        if self.log_transform_step and ds > 0:
            ds_target = np.log(ds + 1e-8)
        else:
            ds_target = ds
        
        # 重启标签
        y_restart = point['y_restart']
        
        return {
            'features': torch.FloatTensor(features),
            'ds_target': torch.FloatTensor([ds_target])[0],
            'y_restart': torch.FloatTensor([y_restart])[0],
            'ds_raw': ds,  # 原始步长 (用于分析)
            'source': point.get('source', 'unknown')
        }
    
    def get_normalization_params(self) -> Dict:
        """返回归一化参数 (用于推理时)"""
        return {
            'feature_mean': self.feature_mean.tolist(),
            'feature_std': self.feature_std.tolist(),
            'ds_mean': float(self.ds_mean),
            'ds_std': float(self.ds_std)
        }


class DataGenerator:
    """
    完整数据生成流水线
    """
    
    def __init__(self, A: np.ndarray, epsilon: float,
                 feature_extractor: callable,
                 svd_solver: callable = None,
                 save_dir: str = 'data/'):
        """
        :param A: 目标矩阵
        :param epsilon: 伪谱水平
        :param feature_extractor: 7 维特征提取函数
        :param svd_solver: SVD 求解器
        :param save_dir: 数据保存目录
        """
        self.A = A
        self.epsilon = epsilon
        self.feature_extractor = feature_extractor
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化专家求解器
        self.expert_solver = ExpertSolver(
            A=A, epsilon=epsilon,
            rtol=1e-8, atol=1e-8,
            drift_threshold=1e-4,
            svd_solver=svd_solver
        )
        
        # 初始化 DAgger 增强器
        self.dagger_augmenter = DAggerAugmenter(
            expert_solver=self.expert_solver,
            feature_extractor=self.feature_extractor
        )
    
    def generate_from_matrix(self, z0: complex,
                              max_steps: int = 500,
                              num_perturbations: int = 3,
                              save: bool = True) -> PseudospectrumDataset:
        """
        从给定矩阵和起点生成完整数据集
        
        :param z0: 等高线起点
        :param max_steps: 最大追踪步数
        :param num_perturbations: 每点扰动样本数
        :param save: 是否保存到磁盘
        :return: PyTorch Dataset
        """
        print(f"=== 生成专家轨迹 (起点 z0={z0}) ===")
        trajectory = self.expert_solver.generate_expert_trajectory(
            z0=z0, max_steps=max_steps
        )
        print(f"专家轨迹长度：{len(trajectory)} 步")
        
        # 统计重启次数
        num_restarts = sum(1 for p in trajectory if p['y_restart'] == 1)
        print(f"SVD 重启次数：{num_restarts}")
        
        print(f"=== DAgger 扰动增强 ===")
        augmented_data = self.dagger_augmenter.augment_trajectory(
            trajectory=trajectory,
            num_perturbations_per_point=num_perturbations
        )
        print(f"增强后数据量：{len(augmented_data)} 样本")
        
        # 统计类别分布
        num_positive = sum(1 for d in augmented_data if d['y_restart'] == 1)
        print(f"重启样本比例：{num_positive}/{len(augmented_data)} = {num_positive/len(augmented_data):.2%}")
        
        # 创建 Dataset
        dataset = PseudospectrumDataset(augmented_data, normalize=True)
        
        if save:
            self._save_dataset(dataset, augmented_data, z0)
        
        return dataset
    
    def generate_multiple_trajectories(self, z0_list: List[complex],
                                        max_steps: int = 500,
                                        num_perturbations: int = 3,
                                        save: bool = True) -> PseudospectrumDataset:
        """
        从多个起点生成多条轨迹，合并为大数据集
        
        :param z0_list: 多个起点列表
        :return: 合并后的 Dataset
        """
        all_data = []
        
        for i, z0 in enumerate(z0_list):
            print(f"\n=== 轨迹 {i+1}/{len(z0_list)}: z0={z0} ===")
            
            trajectory = self.expert_solver.generate_expert_trajectory(
                z0=z0, max_steps=max_steps
            )
            
            augmented_data = self.dagger_augmenter.augment_trajectory(
                trajectory=trajectory,
                num_perturbations_per_point=num_perturbations
            )
            
            all_data.extend(augmented_data)
            print(f"累计数据量：{len(all_data)}")
        
        dataset = PseudospectrumDataset(all_data, normalize=True)
        
        if save:
            self._save_dataset(dataset, all_data, 'multiple')
        
        return dataset
    
    def _save_dataset(self, dataset: PseudospectrumDataset, 
                      raw_data: List[Dict], identifier: str):
        """保存数据集到磁盘"""
        timestamp = np.datetime64('now', 's')
        save_name = f"dataset_{identifier}_{timestamp}".replace('-', '_').replace(':', '_')
        
        # 保存原始数据 (JSON)
        raw_path = self.save_dir / f"{save_name}_raw.json"
        # 序列化复数等特殊类型
        raw_data_serializable = []
        for point in raw_data:
            point_copy = point.copy()
            point_copy['z'] = complex(point_copy['z'])
            point_copy['features'] = point_copy['features'].tolist()
            raw_data_serializable.append(point_copy)
        
        with open(raw_path, 'w') as f:
            json.dump(raw_data_serializable, f, indent=2)
        
        # 保存归一化参数
        norm_params = dataset.get_normalization_params()
        norm_path = self.save_dir / f"{save_name}_norm_params.json"
        with open(norm_path, 'w') as f:
            json.dump(norm_params, f, indent=2)
        
        # 保存 PyTorch Dataset (用 torch.save)
        dataset_path = self.save_dir / f"{save_name}_dataset.pt"
        torch.save(dataset, dataset_path)
        
        print(f"数据已保存到：{self.save_dir}")
        print(f"  - 原始数据：{raw_path}")
        print(f"  - 归一化参数：{norm_path}")
        print(f"  - Dataset: {dataset_path}")
    
    @staticmethod
    def load_dataset(dataset_path: str, 
                     norm_params_path: str) -> PseudospectrumDataset:
        """加载已保存的数据集"""
        dataset = torch.load(dataset_path)
        
        # 加载归一化参数
        with open(norm_params_path, 'r') as f:
            norm_params = json.load(f)
        dataset.feature_mean = np.array(norm_params['feature_mean'])
        dataset.feature_std = np.array(norm_params['feature_std'])
        dataset.ds_mean = norm_params['ds_mean']
        dataset.ds_std = norm_params['ds_std']
        
        return dataset
```

---

## 5. 使用示例

### 5.1 生成单条轨迹数据

```python
# scripts/generate_data_single.py

import numpy as np
from src.core.manifold_ode import extract_features
from src.train.data_generator import DataGenerator

# === 配置 ===
n = 100  # 矩阵规模
epsilon = 0.1  # 伪谱水平

# 生成随机矩阵 (可以是任意矩阵)
np.random.seed(42)
A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
A = A @ A.conj().T / n  # 构造 Hermitian 矩阵 (可选)

# 起点 (需要在伪谱等高线上)
z0 = 1.0 + 0.5j

# === 生成数据 ===
generator = DataGenerator(
    A=A,
    epsilon=epsilon,
    feature_extractor=extract_features,
    save_dir='data/single_trajectory/'
)

dataset = generator.generate_from_matrix(
    z0=z0,
    max_steps=300,
    num_perturbations=3,
    save=True
)

print(f"\n数据集统计:")
print(f"  总样本数：{len(dataset)}")
print(f"  特征维度：{dataset[0]['features'].shape}")

# 创建 DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
print(f"  DataLoader 批次数：{len(dataloader)}")
```

### 5.2 生成多条轨迹数据 (推荐)

```python
# scripts/generate_data_multiple.py

import numpy as np
from src.core.manifold_ode import extract_features
from src.train.data_generator import DataGenerator

# === 配置 ===
n = 100
epsilon = 0.1

# 生成随机矩阵
np.random.seed(42)
A = np.random.randn(n, n) + 1j * np.random.randn(n, n)

# === 选择多个起点 ===
# 方法 1: 在复平面上均匀采样，找到伪谱等高线上的点
# 方法 2: 先运行一次粗略搜索，找到等高线上的多个点

# 这里演示方法 1 (简单版)
def find_point_on_contour(A, epsilon, angle):
    """
    给定角度，找到该方向上伪谱等高线上的点
    通过射线搜索
    """
    from scipy.optimize import brentq
    
    def sigma_minus_epsilon(r):
        z = r * np.exp(1j * angle)
        M = z * np.eye(n) - A
        sigma_min = np.linalg.norm(np.linalg.solve(M, np.random.randn(n)))
        return sigma_min - epsilon
    
    # 二分搜索
    try:
        r_sol = brentq(sigma_minus_epsilon, 0.1, 5.0)
        return r_sol * np.exp(1j * angle)
    except:
        return None

# 生成 8 个起点 (均匀分布在复平面)
z0_list = []
for k in range(8):
    angle = 2 * np.pi * k / 8
    z0 = find_point_on_contour(A, epsilon, angle)
    if z0 is not None:
        z0_list.append(z0)
        print(f"起点 {k}: z0 = {z0:.4f}")

# === 生成数据 ===
generator = DataGenerator(
    A=A,
    epsilon=epsilon,
    feature_extractor=extract_features,
    save_dir='data/multiple_trajectories/'
)

dataset = generator.generate_multiple_trajectories(
    z0_list=z0_list,
    max_steps=200,
    num_perturbations=3,
    save=True
)

print(f"\n最终数据集:")
print(f"  总样本数：{len(dataset)}")
```

### 5.3 数据分析和可视化

```python
# scripts/analyze_data.py

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# 加载数据
with open('data/multiple_trajectories/dataset_raw.json', 'r') as f:
    data = json.load(f)

# === 分析 1: 步长分布 ===
ds_values = [d['ds_expert'] for d in data if d['ds_expert'] > 0]
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(ds_values, bins=50, log=True)
plt.xlabel('Expert Step Size (ds)')
plt.ylabel('Count (log scale)')
plt.title('步长分布直方图')

plt.subplot(1, 2, 2)
plt.hist(np.log(np.array(ds_values) + 1e-8), bins=50)
plt.xlabel('Log(ds)')
plt.title('对数步长分布')
plt.tight_layout()
plt.savefig('analysis_step_size.png')

# === 分析 2: 重启样本分布 ===
y_restart = [d['y_restart'] for d in data]
restart_ratio = sum(y_restart) / len(y_restart)

plt.figure(figsize=(8, 6))
plt.bar(['No Restart', 'Restart'], 
        [len(y_restart) - sum(y_restart), sum(y_restart)])
plt.ylabel('Count')
plt.title(f'重启样本分布 (比例={restart_ratio:.2%})')
plt.savefig('analysis_restart_dist.png')

# === 分析 3: 特征相关性 ===
features_stack = np.array([d['features'] for d in data])
feature_names = ['f1:σ误差', 'f2:u 漂移', 'f3:v 漂移', 'f4:残差', 
                 'f5:梯度模', 'f6:曲率', 'f7:迭代数']

plt.figure(figsize=(10, 8))
corr_matrix = np.corrcoef(features_stack.T)
plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='相关系数')
plt.xticks(range(7), feature_names, rotation=45)
plt.yticks(range(7), feature_names)
plt.title('7 维特征相关系数矩阵')
for i in range(7):
    for j in range(7):
        plt.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center')
plt.tight_layout()
plt.savefig('analysis_feature_corr.png')

print("分析完成，图表已保存")
```

---

## 6. 数据生成最佳实践

### 6.1 矩阵选择策略

| 矩阵类型 | 用途 | 建议 |
|----------|------|------|
| 随机复矩阵 | 基础测试 | `A = randn(n,n) + i*randn(n,n)` |
| 随机 Hermitian | 物理系统模拟 | `A = (R + R')/2` |
| 非正规矩阵 | 测试算法鲁棒性 | `A = randn(n,n)` 不对称 |
| 稀疏矩阵 | 大规模测试 | 用 `scipy.sparse` 生成 |
| 实际应用矩阵 | 最终验证 | 从 Matrix Market 下载 |

### 6.2 起点选择策略

**目标**：覆盖等高线的不同几何特征区域

1. **平缓区域**：曲率小，步长大
2. **尖端区域**：曲率大，步长小
3. **连通分支合并点**：梯度接近 0

**推荐方法**：
- 先用粗网格扫描伪谱等高线
- 在等高线上均匀采样 8-16 个起点
- 确保覆盖所有几何特征区域

### 6.3 数据量建议

| 阶段 | 轨迹数 | 每轨迹步数 | 扰动倍数 | 总样本量 |
|------|--------|------------|----------|----------|
| 原型验证 | 1-2 | 100 | 2 | ~600 |
| 初步训练 | 4-8 | 200 | 3 | ~6400 |
| 充分训练 | 16-32 | 300 | 3 | ~38400 |
| 生产级 | 64+ | 500 | 5 | ~160000+ |

### 6.4 类别不平衡处理

重启样本通常只占 1-5%，需要特殊处理：

```python
# 方法 1: 过采样重启样本
from sklearn.utils import resample

restart_samples = [d for d in data if d['y_restart'] == 1]
non_restart_samples = [d for d in data if d['y_restart'] == 0]

# 过采样重启样本到 20% 比例
restart_samples_oversampled = resample(
    restart_samples,
    replace=True,
    n_samples=len(non_restart_samples) // 4
)

balanced_data = non_restart_samples + restart_samples_oversampled

# 方法 2: 使用加权损失 (在训练时)
# alpha_restart = 0.9 (见 loss.py)
```

---

## 7. 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 专家轨迹太短 (<50 步) | 起点不在等高线上 | 用 Brent 搜索精确起点 |
| 重启比例过高 (>20%) | drift_threshold 太严格 | 放宽到 1e-3 或检查矩阵条件数 |
| 步长分布过于集中 | RK45 容差不够严苛 | 设置 rtol=atol=1e-10 |
| DAgger 样本质量差 | 噪声强度过大 | 减小 noise_scales 参数 |
| 内存不足 | 矩阵太大 + 轨迹太多 | 分批生成，及时保存 |

---

## 8. 数据生成检查清单

在开始训练前，请确认：

- [ ] 专家轨迹长度足够 (每轨迹 >100 步)
- [ ] 重启样本比例在 1-10% 之间
- [ ] 步长分布跨越至少 2 个数量级
- [ ] DAgger 扰动样本数 ≥ 原始样本数的 2 倍
- [ ] 特征归一化参数已保存
- [ ] 数据集已持久化到磁盘
- [ ] 已可视化分析数据分布

---

## 9. 下一步

数据生成完成后，进入训练阶段：

1. 加载 `PseudospectrumDataset`
2. 创建 `DataLoader` (batch_size=128, shuffle=True)
3. 初始化 `NNController` 和 `ControllerLoss`
4. 开始训练 (见 `src/train/trainer.py`)
