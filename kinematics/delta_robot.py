import jax
import jax.numpy as jnp
from functools import partial

class DeltaRobot:
    """
    Delta机器人可微运动学及几何误差模型。
    该类的实现严格参考了论文  中的几何误差建模方法。
    所有计算均使用 JAX 以支持自动微分。
    """
    def __init__(self, params):
        self.Rf = params['fixed_platform_radius']
        self.Re = params['moving_platform_radius']
        self.L = params['active_arm_length']
        self.l = params['passive_arm_length']
        self.g_tool = jnp.array(params['tool_offset'])
        self.identified_params = jnp.zeros(42)

        angles_B = jnp.array([0, 2 * jnp.pi / 3, 4 * jnp.pi / 3])
        self.B = self.Rf * jnp.vstack([jnp.cos(angles_B), jnp.sin(angles_B), jnp.zeros(3)]).T

        angles_A = jnp.array([0, 2 * jnp.pi / 3, 4 * jnp.pi / 3])
        self.A = self.Re * jnp.vstack([jnp.cos(angles_A), jnp.sin(angles_A), jnp.zeros(3)]).T

    @partial(jax.jit, static_argnums=(0,))
    def inverse_kinematics(self, P):
        thetas = []
        for i in range(3):
            angle = i * 2 * jnp.pi / 3
            R_z = jnp.array([[jnp.cos(angle), -jnp.sin(angle), 0],
                             [jnp.sin(angle), jnp.cos(angle), 0],
                             [0, 0, 1]])
            p_rot = R_z.T @ P
            a_rot = R_z.T @ self.A[i]
            b_rot = R_z.T @ self.B[i]
            x, y, z = p_rot
            E = 2 * self.L * (x - b_rot[0])
            F = 2 * self.L * z
            G = (x - b_rot[0])**2 + y**2 + z**2 + self.L**2 - self.l**2
            sqrt_val = jnp.sqrt(E**2 + F**2 - G**2)
            theta = 2 * jnp.arctan2(F + sqrt_val, E + G)
            thetas.append(theta)
        return jnp.array(thetas)

    @partial(jax.jit, static_argnums=(0,))
    def forward_kinematics_nominal(self, thetas):
        C = jnp.zeros((3, 3))
        for i in range(3):
            angle = i * 2 * jnp.pi / 3
            R_z = jnp.array([[jnp.cos(angle), -jnp.sin(angle), 0],
                             [jnp.sin(angle), jnp.cos(angle), 0],
                             [0, 0, 1]])
            theta_i = thetas[i] if thetas.ndim > 0 else thetas
            c_local = jnp.array([self.L * jnp.cos(theta_i), 0.0, -self.L * jnp.sin(theta_i)])
            C = C.at[i].set(self.B[i] + R_z @ c_local)

        def objective(P_guess):
            errs = jnp.sum((P_guess - C)**2, axis=1) - self.l**2
            return jnp.sum(errs**2)

        grad_fn = jax.grad(objective)
        P_nominal = jnp.array([0.0, 0.0, -0.8])
        for _ in range(20):
            grad = grad_fn(P_nominal)
            P_nominal -= grad * 0.05
        return P_nominal

    @partial(jax.jit, static_argnums=(0,))
    def forward_kinematics_with_errors(self, thetas, delta_p):
        P_nominal = self.forward_kinematics_nominal(thetas)
        A_rr, A_re, A_ee = jnp.zeros((3, 3)), jnp.zeros((3, 3)), jnp.zeros((3, 3))
        B_r, B_e = jnp.zeros((3, 24)), jnp.zeros((3, 18))

        for i in range(3):
            angle = i * 2 * jnp.pi / 3
            R_z = jnp.array([[jnp.cos(angle), -jnp.sin(angle), 0],
                             [jnp.sin(angle), jnp.cos(angle), 0],
                             [0, 0, 1]])
            theta_i = thetas[i] if thetas.ndim > 0 else thetas
            c_local = jnp.array([self.L * jnp.cos(theta_i), 0.0, -self.L * jnp.sin(theta_i)])
            C_i = self.B[i] + R_z @ c_local
            l_i_vec = P_nominal + self.A[i] - C_i
            l_i_norm = jnp.linalg.norm(l_i_vec)
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

        dp_r = jnp.concatenate([delta_p[i*14 : i*14+8] for i in range(3)])
        dp_e = jnp.concatenate([delta_p[i*14+8 : (i+1)*14] for i in range(3)])
        A_ee_inv = jnp.linalg.pinv(A_ee)
        delta_e = A_ee_inv @ (B_e @ dp_e)
        A_rr_inv = jnp.linalg.pinv(A_rr)
        delta_r = A_rr_inv @ (B_r @ dp_r - A_re @ delta_e)
        da, db, dg = delta_e
        R = jnp.array([[1., -dg, db], [dg, 1., -da], [-db, da, 1.]])
        delta_g = delta_r + R @ self.g_tool - self.g_tool
        P_actual = P_nominal + delta_g
        return P_actual