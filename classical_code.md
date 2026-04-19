# 伪谱曲线追踪算法 (Pseudospectrum Curve Tracing) 实现指南

> 说明：这个文件记录的是论文/传统 predictor-corrector 的参考算法，不对应当前仓库实现。当前实现请看 `README.md`、`NN_8维输入梳理.md` 和 `算法完整流程与单步计算全过程.md`。

## 1. 核心数学公式推导（面向代码实现）

我们的目标是追踪复平面上的等高线：$f(z) = \sigma_{min}(zI - A) - \epsilon = 0$。
令复数 $z = x + iy$，矩阵 $M(z) = zI - A$。

设在点 $z$ 处，$M(z)$ 的最小奇异值为 $\sigma$，对应的左、右奇异向量分别为 $u$ 和 $v$：
$$ M(z) v = \sigma u \quad \text{且} \quad M(z)^H u = \sigma v $$

根据矩阵微扰理论，最小奇异值对 $x, y$ 的梯度可以非常精妙地转化为复数运算：
*   **计算复梯度 (Complex Gradient)**：
    定义复数 $g(z) = v^H u$ （即 $v$ 和 $u$ 的内积）。
    梯度的实部为 $Re(g)$，虚部为 $Im(g)$。所以在程序中，我们直接使用标量复数 $g$ 来代表梯度。
*   **计算切线方向 (Tangent Direction)**：
    将梯度逆时针旋转 90 度，即乘以单位虚数 $i$。
    归一化后的切线方向（复数）为： $\tau = i \frac{g}{|g|}$
*   **牛顿校正步 (Newton Corrector)**：
    沿梯度方向进行单步一维牛顿法，将偏离的预测点拉回等高线。复数位移公式为 $\Delta z_{corr} = - \frac{\sigma - \epsilon}{|g|^2} \cdot g$ 

---

## 2. 算法输入与输出定义

**输入 (Inputs)**:
*   `A`: 给定的 $n \times n$ 矩阵（通常为复矩阵）。
*   `epsilon` ($\epsilon$): 目标伪谱边界的容差/层级（标量）。
*   `z0`: 初始起始点（复数），必须**严格位于** $\sigma_{min}(z_0I - A) \approx \epsilon$ 的曲线上。
*   `h`: 初始追踪步长（标量，例如 0.05）。
*   `max_steps`: 最大允许的迭代步数，防止死循环。

**输出 (Outputs)**:
*   `curve_points`: 一系列复数坐标数组 `[z_0, z_1, ..., z_n]`，连接起来构成闭合的伪谱边界曲线。

---

## 3. 标准伪代码实现流程

```python
# ==============================================================================
# Helper Function: 计算最小奇异值及其向量
# ==============================================================================
function get_smallest_svd(A, z, u_guess=None, v_guess=None):
    M = z * I - A
    
    # 工业界实现注意：
    # 对于大型稀疏矩阵，绝对不要算全量SVD。
    # 应该使用逆迭代法(Inverse Iteration)或 Lanczos双斜边法 求最小奇异值。
    # u_guess 和 v_guess 可作为 krylov 子空间的暖启动向量(Warm Start)，极大加速收敛。
    
    sigma, u, v = Compute_Min_SVD(M, u_guess, v_guess)
    return sigma, u, v

# ==============================================================================
# Main Algorithm: 预测-校正 曲线追踪
# ==============================================================================
function trace_pseudospectrum(A, epsilon, z0, h=0.01, max_steps=5000):
    
    curve_points = [z0]
    z_k = z0
    
    # 获取初始点的状态
    sigma_k, u_k, v_k = get_smallest_svd(A, z_k)
    g_k = dot_product(conjugate_transpose(v_k), u_k) # 计算复梯度 g = v^H * u
    
    # 定义追踪方向参数 (+1 或 -1 控制顺时针/逆时针)
    direction = 1 
    
    for step in 1 to max_steps:
        
        # ---------------------------------------------------------
        # [步骤 1] 预测步 (Predictor)
        # 沿着当前点的切线方向，以前进长度 h 探索下一个点
        # ---------------------------------------------------------
        tangent = direction * 1j * g_k / abs(g_k)  # 1j 为虚数单位 i
        z_pred = z_k + h * tangent
        
        # ---------------------------------------------------------
        # [步骤 2] 校正步 (Corrector)
        # 在预测点评估偏差，并使用 1 步 Newton 法垂直拉回到真实曲线上
        # ---------------------------------------------------------
        sigma_p, u_p, v_p = get_smallest_svd(A, z_pred, u_guess=u_k, v_guess=v_k)
        g_p = dot_product(conjugate_transpose(v_p), u_p)
        
        # 牛顿法步长位移
        delta_z = - (sigma_p - epsilon) / (abs(g_p)^2) * g_p
        
        # 得到校正后的新点
        z_next = z_pred + delta_z
        
        # ---------------------------------------------------------
        # [步骤 3] 步长自适应 (Step-size Adaptive Control) Optional
        # ---------------------------------------------------------
        # 评估牛顿位移长如果太大，说明预测点偏离太多，曲率较大，应当减少步长重新算
        if abs(delta_z) > (0.5 * h):
            h = h * 0.5
            continue # 放弃当前 z_next，重新用小步长走当前循环
            
        # 如果拟合得极好，可以适当放大步长，加速在平缓区域的追踪
        if abs(delta_z) < (0.05 * h):
            h = min(h * 1.2, h_max)
            
        # ---------------------------------------------------------
        # [步骤 4] 接受该点并更新状态
        # ---------------------------------------------------------
        z_k = z_next
        u_k = u_p
        v_k = v_p
        g_k = g_p # 虽然标准的牛顿法应该用矫正后的点再算一次梯度，但近似用预测点的即可
        
        curve_points.append(z_k)
        
        # ---------------------------------------------------------
        # [步骤 5] 闭环/终止检查 (Termination Check)
        # ---------------------------------------------------------
        # 如果走过了初始的盲区，并且当前点回到了起点 z0 附近，说明曲线成功闭合
        if step > 10 and abs(z_k - z0) < h:
            break
            
    return curve_points