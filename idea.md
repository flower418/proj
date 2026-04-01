# 动机

* 传统算法（如牛顿预估矫正）每走一步都需要计算 $\sigma,u,v$，开销较大，于是可以利用子空间的一些性质，将 SVD 进行微分传播
    * 只需要已知点的 $(z,u,v)$，就能列出随弧长 $s$ 变化的微分方程
    * 在更新 $z$ 的同时，$(u,v)$ 可以通过微分进行传播
    * 积分 ODE 只需要 $O(n^2)$
* 纯 ODE 在进行多步后会产生数值漂移，需要进行一次 SVD “重启”，传统做法要人工设定在什么条件下进行 SVD，会导致大量不必要的 SVD，可以引入 NN 来优化这一问题
    * 训练一个轻量级的网络
    * 输入：当前的局部几何特征
    * 输出：
        * 最优的步长 $ds$：在平缓区域 $ds$ 较大，在尖端区域 $ds$ 较小
        * 是否需要重启：判断线性近似是否已经误差过大，必须做一次 SVD



# 推导

## 1.问题设定与基础约束

给定大矩阵 $A \in \mathbb{C}^{n \times n}$，定义含参矩阵 $M(z) = zI - A$。对于伪谱水平 $\epsilon > 0$，等高线 $\Lambda_\epsilon(A)$ 上的任意一点 $z$，其最小奇异值满足 $\sigma(z) = \epsilon$。

设 $(\sigma, u, v)$ 为 $M(z)$ 的最小奇异三元组，它们必须满足以下基本定义：
$$
M(z)v = \sigma u \quad \text{(1a)} 
$$
$$
M(z)^*u = \sigma v \quad \text{(1b)} 
$$
同时，左、右奇异向量满足正交归一化约束：
$$
u^*u = 1, \quad v^*v = 1 \quad \text{(2)} 
$$

假设 $z(s) = x(s) + i y(s)$ 是沿着伪谱等高线参数化的目标曲线，其中 $s$ 为弧长参数。我们定义对弧长 $s$ 的导数为 $\dot{(\cdot)} = \frac{d}{ds}$。由于等高线上奇异值恒定，我们有首要约束条件：
$$
\dot{\sigma} = 0 \quad \text{(3)} 
$$

## 2.预测轨线：等高线切向的严谨推导

为了追踪等高线，我们需要首先确定复平面上的移动方向 $\dot{z} = \frac{dz}{ds}$。

由于 $\frac{dM}{ds} = \dot{z}I$ 且 $\frac{dM^*}{ds} = \bar{\dot{z}}I$，我们对式 (1a) 和 (1b) 两边同时对 $s$ 求导，并代入 $\dot{\sigma} = 0$ 得到：
$$
\dot{z}v + M\dot{v} = \sigma\dot{u} \quad \text{(4a)} 
$$
$$
\bar{\dot{z}}u + M^*\dot{u} = \sigma\dot{v} \quad \text{(4b)} 
$$

为了解出 $\dot{z}$，我们在式 (4a) 两侧左乘 $u^*$：
$$
u^*(\dot{z}v) + u^*M\dot{v} = \sigma u^*\dot{u} 
$$
利用关系 $u^*M = \sigma v^*$（由 1b 的共轭转置得到），上式化简为：
$$
\dot{z}(u^*v) + \sigma v^*\dot{v} = \sigma u^*\dot{u} \quad \text{(5)} 
$$

**引入规范条件 (Gauge Condition)：** 
在复奇异值分解中，奇异向量存在一个纯相位的自由度（即乘以 $e^{i\theta}$ 依然满足定义）。为了在连续演化中固定这一自由度，确保数值稳定性，我们引入最小相位变化规范条件：
$$
\text{Re}(v^*\dot{v}) = 0 \quad \text{和} \quad \text{Re}(u^*\dot{u}) = 0 \quad \text{(6)} 
$$
对式 (5) 两边取实部，并应用该规范条件，立即可得：
$$
\text{Re}(\dot{z} u^*v) = 0 \quad \text{(7)} 
$$

**切向方程的几何意义：**
令复数量 $\gamma = u^*v$（这实质上代表了 $\sigma(z)$ 在复平面上的复梯度）。式 (7) 表明，我们的运动方向 $\dot{z}$ 必须与梯度 $\gamma$ 在复平面上正交。为了保证单位弧长推进（$|\dot{z}|=1$），我们对梯度旋转 $90^\circ$ （乘以虚数单位 $i$）并归一化，得到等高线的完美切向方程：
$$
\frac{dz}{ds} = i \frac{\bar{\gamma}}{|\gamma|} = i \frac{v^*u}{|v^*u|} \quad \text{(8)} 
$$

## 3.偏导数的解耦与伪逆映射

明确了前进方向 $\dot{z}$ 后，核心挑战在于：**如何不调用 SVD 也能同步更新高维奇异向量 $u$ 和 $v$？** 

我们回到耦合的变分方程组 (4a) 和 (4b)：
$$
\sigma\dot{u} - M\dot{v} = \dot{z}v \quad \text{(9a)} 
$$
$$
M^*\dot{u} - \sigma\dot{v} = -\bar{\dot{z}}u \quad \text{(9b)} 
$$

为了解耦 $\dot{v}$，我们在 (9a) 左侧乘以 $M^*$，在 (9b) 左侧乘以 $\sigma$：
$$
\sigma M^*\dot{u} - M^*M\dot{v} = \dot{z}M^*v 
$$
$$
\sigma M^*\dot{u} - \sigma^2 \dot{v} = -\sigma\bar{\dot{z}}u 
$$
两式相减，成功消去 $\dot{u}$，得到关于 $\dot{v}$ 的方程：
$$
(M^*M - \sigma^2 I)\dot{v} = -(\dot{z}M^*v + \sigma\bar{\dot{z}}u) \quad \text{(10)} 
$$

同理，为了解耦 $\dot{u}$，在 (9b) 左侧乘以 $M$，在 (9a) 左侧乘以 $\sigma$，相减可得：
$$
(MM^* - \sigma^2 I)\dot{u} = -(\bar{\dot{z}}Mu + \sigma\dot{z}v) \quad \text{(11)} 
$$

**奇异性与摩尔-彭若斯伪逆 (Moore-Penrose Pseudoinverse)：**
注意到算子 $(M^*M - \sigma^2 I)$ 是奇异的，因为由定义可知 $v$ 恰好位于它的零空间中（$(M^*M - \sigma^2 I)v = 0$）。然而，经过推导可以证明，方程 (10) 的右侧向量恰好与 $v$ 正交（即位于值域内）。因此，该线性方程系统存在一致解。
我们使用摩尔-彭若斯伪逆 $(\cdot)^+$，可以求出唯一且天然垂直于 $v$ 的极小范数解，这完美契合了我们的正交向演化需求。

由此，我们得出了左、右奇异向量针对弧长 $s$ 的严格偏微分方程：
$$
\frac{dv}{ds} = -(M^*M - \epsilon^2 I)^+ \left( \frac{dz}{ds} M^*v + \epsilon \frac{d\bar{z}}{ds} u \right) \quad \text{(12a)} 
$$
$$
\frac{du}{ds} = -(MM^* - \epsilon^2 I)^+ \left( \frac{d\bar{z}}{ds} Mu + \epsilon \frac{dz}{ds} v \right) \quad \text{(12b)} 
$$

## 4.完整微分流形系统

综上所述，我们将寻找伪谱等高线的全局离散搜索映射为一个完全确定的连续常微分方程初值问题（IVP）。完整的白盒底座系统如下：

$$
\begin{cases}
  \frac{dz}{ds} = i \frac{v^*u}{|v^*u|}  \\[10pt]
  \frac{dv}{ds} = -\big(M^*M - \epsilon^2 I\big)^+ \big( \dot{z} M^*v + \epsilon \bar{\dot{z}} u \big) \\[10pt]
  \frac{du}{ds} = -\big(MM^* - \epsilon^2 I\big)^+ \big( \bar{\dot{z}} Mu + \epsilon \dot{z} v \big)
\end{cases} \quad \text{(13)}
$$



# 伪代码

```python
import numpy as np
import torch

def neural_subspace_tracking(A, epsilon, z0, nn_controller, max_steps=1000):
    """
    神经增强的子空间追踪算法
    :param A: 巨大的目标矩阵 (n x n)
    :param epsilon: 我们要求解的伪谱等高线水平 (例如 0.1)
    :param z0: 等高线上的一个初始起点 (复数)
    :param nn_controller: 我们训练好的神经网络
    """
    
    # ==========================================
    # 1. 初始化（无可避免地做一次昂贵的完整 SVD）
    # ==========================================
    # 找到 z0 处精确的最小奇异对 (u, v)
    z = z0
    u, v = exact_SVD_computation(A, z, epsilon)  # 复杂度 O(n^3)
    
    trajectory = [z] # 记录轨迹
    
    # ==========================================
    # 2. 核心追踪循环（沿着等高线狂飙）
    # ==========================================
    for step in range(max_steps):
        
        # 步骤 A：提取当前环境状态 (构建 NN 的输入)
        # 包括：梯度大小 |u*v|，约束误差 ||u||-1，局部曲率等
        state_features = extract_geometrical_state(z, u, v, A, epsilon)
        
        # 步骤 B：神经网络(大脑)下达指令！
        # NN 根据当前地形，输出【最佳步长】和【是否需要求援(SVD)】
        ds, need_restart = nn_controller(state_features)
        
        # ---------------------------------------------------
        # 步骤 C：执行动作（算法的核心分水岭！）
        # ---------------------------------------------------
        if need_restart == True:
            # 【AI 发现危险】：漂移太严重，或者遇到了奇点/急转弯
            # 必须踩刹车，调用极其昂贵但精确的传统 SVD 进行“洗牌”校正
            z, u, v = exact_SVD_computation(A, z, epsilon)  # 复杂度 O(n^3) 😱
            
        else:
            # 【AI 认为安全】：一切在掌控中，用我们推导的微分方程飞速滑行！
            # 计算导数 (调用前文推导的数学公式)
            dz_ds, du_ds, dv_ds = compute_manifold_derivatives(z, u, v, A, epsilon)
            
            # 走一步 (这里用最简单的欧拉法示意，实际应用RK4)
            z = z + ds * dz_ds
            u = u + ds * du_ds
            v = v + ds * dv_ds
            
            # 顺手把 u, v 拉回单位圆 (极低成本的正则化)
            u = u / np.linalg.norm(u)
            v = v / np.linalg.norm(v)
            # 复杂度仅仅是几次矩阵向量乘积 O(n^2) 🚀
            
        trajectory.append(z)
        
        # 步骤 D：闭合检测
        # 如果 z 绕了一圈回到了起点 z0 附近，等高线画完了！
        if is_closed(z, z0) and step > 10:
            print(f"追踪完成！总步数: {step}")
            break
            
    return trajectory
```

# 网络设计

## 1.状态空间设计：维度无关的标量不变量 

**网络的输入必须严格与矩阵的具体维度解耦**。
在追踪的第 $t$ 步，我们提取当前状态 $(z_t, u_t, v_t)$ 的局部几何形貌与数值健康度，构建一个固定维度（如 7 维）的标量特征向量 $X_t \in \mathbb{R}^7$

*   **数值漂移群 (Drift Indicators):**
    *   $f_1 = | \text{Re}(u_t^* M_t v_t) - \epsilon |$：当前近似奇异值与目标等高线水平 $\epsilon$ 的绝对偏差（无需全量 SVD，通过 Rayleigh 商极低成本近似）。
    *   $f_2 = |1 - \|u_t\|_2|, \quad f_3 = |1 - \|v_t\|_2|$：奇异向量数值漂移导致的正交约束/模长流失率。
    *   $f_4 = \|M_t v_t - \epsilon u_t\|_2$：残差范数，反映当前 $u,v$ 逼近真实奇异向量的置信度。
*   **局部几何群 (Geometric Topography):**
    *   $f_5 = |u_t^* v_t|$：复梯度的模长。该值接近 0 时意味着伪谱等高线曲率急剧变化或面临拓扑分岔（如多个伪谱连通分支的合并点）。
    *   $f_6 = |\arg(u_t^* v_t) - \arg(u_{t-1}^* v_{t-1})|$：切向方向的变化率，显式提供局部流形的曲率估计。
*   **算力健康度 (Solver Health):**
    *   $f_7 = N_{iter}$：上一积分步中，求解伪逆线性系统（如式 12）时 Krylov 求解器（如 MINRES/PCG）所消耗的迭代次数。迭代次数的异常飙升是矩阵接近奇异、算力即将崩溃的强烈前兆。

## 2.动作空间与网络架构 (Action Space & Architecture)

元控制器设计为一个轻量级的多层感知机（MLP），包含共享的隐藏层特征提取模块，并在最后一层分为两个分支头部（Two-Headed Output）：

1.  **自适应步长头部 (Step Size Head, 回归任务):**
    输出 $\Delta s_{t}$ 经过 `Softplus` 激活，确保预测步长严格为正。
2.  **重启决策头部 (Restart Decision Head, 分类任务):**
    输出概率 $P_{restart} \in (0, 1)$ 经过 `Sigmoid` 激活。当 $P_{restart} > 0.5$ 时，算法将主动中止 ODE 推进，调用一次具有绝对精度的 $O(n^3)$ SVD 重置当前状态 $(u_t, v_t)$，以消除累积误差。

## 3.专家数据的生成与 DAgger 策略 (Dataset Aggregation)

我们采用行为克隆（Behavioral Cloning）训练该网络。训练所用的“专家神谕 (Expert Oracle)”是一个设置了极其严苛容差（如 `rtol=1e-8, atol=1e-8`）的高阶 Runge-Kutta 自适应求解器。
数据生成流程包含以下关键设计：
*   **收集理想轨迹：** 专家求解器在面临截断误差增大时会自动缩小步长。我们将专家最终采纳的最大安全步长记作 Ground-Truth $\Delta s_{expert}$。如果残差 $f_4$ 突破预设的安全阈值，专家将被迫执行 SVD，此时标记该步真实标签 $y_{expert} = 1$，反正为 $0$。
*   **状态扰动增强 (DAgger):** 仅在“完美无瑕”的轨迹上训练，会导致网络在推理时面临“复合误差（Covariate Shift）”时不知所措。因此，我们在专家轨迹的状态中人为注入高斯扰动（模拟 ODE 的数值漂移），并记录专家在“糟糕状态”下的恢复行为（如立即断言要求 Restart 或极度缩小步长）。

## 4.损失函数

网络的总体损失函数 $\mathcal{L}_{total}$ 为回归损失与分类损失的加权和：
$$ \mathcal{L}_{total} = \lambda_1 \mathcal{L}_{step} + \lambda_2 \mathcal{L}_{restart} $$

1.  **对数均方误差 (Log-MSE for Step Size):** 
    由于伪谱边界在平缓区域与奇异点附近的最佳步长可能横跨多个数量级（如 $0.1$ 到 $0.0001$），直接使用 MSE 会导致大步长主导梯度。因此我们采用 Log-MSE：
    $$ \mathcal{L}_{step} = \frac{1}{B} \sum_{i=1}^B \left( \log(\Delta s_{pred}^{(i)}) - \log(\Delta s_{expert}^{(i)}) \right)^2 $$
2.  **加权二元交叉熵 (Weighted BCE for Restart):**
    在整个追踪过程中，需要触发全量 SVD 纠偏的时刻是极小概率事件（存在严重的样本不平衡）。为此我们使用加权交叉熵（或 Focal Loss）对少数类（$y_{exp} = 1$）施加更高惩罚：
    $$ \mathcal{L}_{restart} = -\frac{1}{B} \sum_{i=1}^B \bigg[ \alpha y_{exp}^{(i)} \log(P_{restart}^{(i)}) + (1-\alpha) (1 - y_{exp}^{(i)}) \log(1 - P_{restart}^{(i)}) \bigg] $$
