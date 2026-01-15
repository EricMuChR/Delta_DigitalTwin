import jax
import jax.numpy as jnp
from functools import partial

class DeltaRobot:
    """
    Delta机器人数字孪生模型 (V6.0 - 42参数修正版).
    1. 修正了之前版本忽略动平台半径(Re)的几何错误。
    2. 采用解析法正运动学，替代不稳定的梯度下降法。
    3. 仅保留42个几何参数，符合您的基准需求。
    """
    def __init__(self, params):
        self.Rf = params['fixed_platform_radius']
        self.Re = params['moving_platform_radius']
        self.L = params['active_arm_length']
        self.l = params['passive_arm_length']
        self.g_tool = jnp.array(params['tool_offset'])
        
        # 严格遵守：仅42个几何参数
        self.identified_params = jnp.zeros(42)

        # 预计算基座(B)和动平台(A)的名义坐标
        angles = jnp.array([0, 2 * jnp.pi / 3, 4 * jnp.pi / 3])
        self.B = self.Rf * jnp.vstack([jnp.cos(angles), jnp.sin(angles), jnp.zeros(3)]).T
        self.A = self.Re * jnp.vstack([jnp.cos(angles), jnp.sin(angles), jnp.zeros(3)]).T

    @partial(jax.jit, static_argnums=(0,))
    def inverse_kinematics(self, P):
        """解析逆运动学 (修正了动平台半径影响)"""
        thetas = []
        EPS = 1e-9
        
        for i in range(3):
            angle = i * 2 * jnp.pi / 3
            sin_angle = jnp.sin(angle)
            cos_angle = jnp.cos(angle)
            
            # 1. 将 P 转到第 i 支链的局部坐标系
            x_local = P[0] * cos_angle + P[1] * sin_angle
            y_local = -P[0] * sin_angle + P[1] * cos_angle
            z_local = P[2]
            
            # 2. 关键修正：引入 Re 和 Rf 的径向差
            # 几何闭环方程：(val_x - L*cos(th))^2 + val_y^2 + (val_z + L*sin(th))^2 = l^2
            val_x = x_local + self.Re - self.Rf
            val_y = y_local 
            val_z = z_local
            
            # 3. 构造三角方程系数
            E = -2.0 * self.L * val_x
            F = 2.0 * self.L * val_z
            G = val_x**2 + val_y**2 + val_z**2 + self.L**2 - self.l**2
            
            delta = E**2 + F**2 - G**2
            sqrt_val = jnp.sqrt(jnp.maximum(delta, EPS))
            
            # 4. 求解关节角 (选择 Elbow-Down 分支)
            # 公式推导表明 (-F - sqrt) / (G - E) 对应标准下探姿态
            theta = 2.0 * jnp.arctan2(-F - sqrt_val, G - E)
            thetas.append(theta)
            
        return jnp.array(thetas)

    def inverse_kinematics_batch(self, P_batch):
        """批量逆解辅助函数"""
        return jax.vmap(self.inverse_kinematics)(jnp.array(P_batch))

    @partial(jax.jit, static_argnums=(0,))
    def forward_kinematics_nominal(self, thetas):
        """解析正运动学 (三球交点法，无迭代，极快且稳定)"""
        EPS = 1e-9
        C = jnp.zeros((3, 3))
        for i in range(3):
            angle = i * 2 * jnp.pi / 3
            R_z = jnp.array([[jnp.cos(angle), -jnp.sin(angle), 0],
                             [jnp.sin(angle), jnp.cos(angle), 0],
                             [0, 0, 1]])
            c_local = jnp.array([self.L * jnp.cos(thetas[i]), 0.0, -self.L * jnp.sin(thetas[i])])
            # 球心位置 C'_i = C_i - A_i (等效变换)
            C = C.at[i].set(self.B[i] + R_z @ c_local - self.A[i])

        # 提取三球球心
        x1, y1, z1 = C[0]
        x2, y2, z2 = C[1]
        x3, y3, z3 = C[2]
        
        # 构造两两相减的平面方程 M[x,y]^T = V
        w1 = x1**2 + y1**2 + z1**2
        w2 = x2**2 + y2**2 + z2**2
        w3 = x3**2 + y3**2 + z3**2
        
        a11 = 2*(x1-x2); a12 = 2*(y1-y2)
        a21 = 2*(x1-x3); a22 = 2*(y1-y3)
        b1_const = w1-w2; b1_z = -2*(z1-z2)
        b2_const = w1-w3; b2_z = -2*(z1-z3)
        
        # 克拉默法则求解 2x2 线性方程
        det = a11 * a22 - a12 * a21
        det_safe = jnp.where(jnp.abs(det) < 1e-6, 1e-6, det)
        inv_det = 1.0 / det_safe
        
        m_inv_11, m_inv_12 = a22 * inv_det, -a12 * inv_det
        m_inv_21, m_inv_22 = -a21 * inv_det, a11 * inv_det
        
        x_const = m_inv_11 * b1_const + m_inv_12 * b2_const
        x_z     = m_inv_11 * b1_z     + m_inv_12 * b2_z
        y_const = m_inv_21 * b1_const + m_inv_22 * b2_const
        y_z     = m_inv_21 * b1_z     + m_inv_22 * b2_z
        
        # 代入球方程解一元二次方程
        a_sq = x_z**2 + y_z**2 + 1.0
        b_sq = 2*(x_z*(x_const - x1) + y_z*(y_const - y1) - z1)
        c_sq = (x_const - x1)**2 + (y_const - y1)**2 + z1**2 - self.l**2
        
        delta = b_sq**2 - 4*a_sq*c_sq
        sqrt_delta = jnp.sqrt(jnp.maximum(delta, EPS))
        
        # Delta机器人通常工作在下方 (Z更小)
        z_sol = (-b_sq - sqrt_delta) / (2*a_sq)
        x_sol = x_z * z_sol + x_const
        y_sol = y_z * z_sol + y_const
        
        return jnp.array([x_sol, y_sol, z_sol])

    @partial(jax.jit, static_argnums=(0,))
    def forward_kinematics_with_errors(self, thetas_cmd, params_42):
        """
        带几何误差的正运动学 (42参数).
        完全兼容您现有的接口。
        """
        # 1. 计算名义位置 (使用快速解析法)
        q_act = thetas_cmd 
        P_nom_act = self.forward_kinematics_nominal(q_act)
        
        # 2. 几何参数注入 (42参数映射)
        # 为防止奇异，使用加固的求逆
        def safe_inv_3x3(M): return jnp.linalg.inv(M + jnp.eye(3)*1e-9)

        A_rr, A_re, A_ee = jnp.zeros((3, 3)), jnp.zeros((3, 3)), jnp.zeros((3, 3))
        B_r, B_e = jnp.zeros((3, 24)), jnp.zeros((3, 18))

        for i in range(3):
            angle = i * 2 * jnp.pi / 3
            R_z = jnp.array([[jnp.cos(angle), -jnp.sin(angle), 0],
                             [jnp.sin(angle), jnp.cos(angle), 0],
                             [0, 0, 1]])
            theta_i = q_act[i]
            c_local = jnp.array([self.L * jnp.cos(theta_i), 0.0, -self.L * jnp.sin(theta_i)])
            C_i = self.B[i] + R_z @ c_local
            
            l_i_vec = P_nom_act + self.A[i] - C_i
            l_i_norm = jnp.linalg.norm(l_i_vec) + 1e-9
            l_i_hat = l_i_vec / l_i_norm
            L_i_hat = (C_i - self.B[i]) / self.L
            
            w0_i = jnp.array([-jnp.sin(angle), jnp.cos(angle), 0])
            u0_i = jnp.array([jnp.cos(angle), jnp.sin(angle), 0])
            v0_i = jnp.cross(w0_i, u0_i)
            
            A_rr = A_rr.at[i].set(l_i_hat)
            A_re = A_re.at[i].set(jnp.cross(self.A[i], l_i_hat))
            
            br_row = jnp.zeros(24)
            br_row = br_row.at[i*8+0].set(jnp.dot(L_i_hat, l_i_hat))
            br_row = br_row.at[i*8+1:i*8+4].set(-l_i_hat)
            br_row = br_row.at[i*8+4].set(jnp.dot(L_i_hat, l_i_hat))
            br_row = br_row.at[i*8+5].set(jnp.dot(u0_i, l_i_hat))
            br_row = br_row.at[i*8+6].set(jnp.dot(v0_i, l_i_hat))
            br_row = br_row.at[i*8+7].set(1.0)
            B_r = B_r.at[i].set(br_row)
            
            A_ee = A_ee.at[i].set(jnp.cross(l_i_vec, w0_i))
            
            u2_i, v2_i = jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0])
            u3_i, v3_i = jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0])
            
            be_row = jnp.zeros(18)
            be_row = be_row.at[i*6+1].set(jnp.dot(l_i_vec, w0_i))
            be_row = be_row.at[i*6+2].set(-jnp.dot(l_i_vec, v3_i))
            be_row = be_row.at[i*6+3].set(jnp.dot(l_i_vec, u3_i))
            be_row = be_row.at[i*6+4].set(-self.l * jnp.dot(v2_i, w0_i))
            be_row = be_row.at[i*6+5].set(self.l * jnp.dot(u2_i, w0_i))
            B_e = B_e.at[i].set(be_row)

        dp_r = jnp.concatenate([params_42[i*8 : (i+1)*8] for i in range(3)]) 
        dp_e = jnp.concatenate([params_42[24 + i*6 : 24 + (i+1)*6] for i in range(3)]) 
        
        delta_e = safe_inv_3x3(A_ee) @ (B_e @ dp_e)
        delta_r = safe_inv_3x3(A_rr) @ (B_r @ dp_r - A_re @ delta_e)
        
        da, db, dg = delta_e
        R_para = jnp.array([[1., -dg, db], [dg, 1., -da], [-db, da, 1.]])
        delta_g_tool = (R_para - jnp.eye(3)) @ self.g_tool
        
        return P_nom_act + delta_r + delta_g_tool