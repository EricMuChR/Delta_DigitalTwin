import pickle
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

class RealTimeCompensator:
    def __init__(self, robot_model, surrogate_model_path):
        print("初始化多策略实时补偿器...")
        self.robot = robot_model
        try:
            with open(surrogate_model_path, 'rb') as f:
                self.surrogate_model = pickle.load(f)
            print("代理模型加载成功。")
        except FileNotFoundError:
            print(f"警告: 代理模型文件 {surrogate_model_path} 未找到。离线补偿策略将不可用。")
            self.surrogate_model = None
        print("补偿器已就绪。")

    def compensate_online_accuracy_first(self, command_position):
        command_pos_jnp = jnp.array(command_position)
        
        def objective_func(p_comp_np):
            p_comp_jnp = jnp.array(p_comp_np)
            theta = self.robot.inverse_kinematics(p_comp_jnp)
            p_actual = self.robot.forward_kinematics_with_errors(theta, self.robot.identified_params)
            error = jnp.sum((p_actual - command_pos_jnp)**2)
            return np.array(error, dtype=np.float64)

        jac_fn = jax.jacfwd(objective_func)
        def scipy_jac_func(p_comp_np):
            return np.array(jac_fn(jnp.array(p_comp_np)), dtype=np.float64)

        result = minimize(objective_func, 
                          x0=np.array(command_position), 
                          method='L-BFGS-B', 
                          jac=scipy_jac_func,
                          tol=1e-7)
        
        if not result.success:
            print(f"警告: 在线补偿优化未收敛: {result.message}")
        return result.x

    def compensate_offline_speed_first_pointwise(self, command_position):
        if self.surrogate_model is None:
            raise RuntimeError("代理模型未加载，无法执行离线补偿。")
        command_pos_reshaped = np.array(command_position).reshape(1, -1)
        predicted_error = self.surrogate_model.predict(command_pos_reshaped)
        compensated_position = command_pos_reshaped - predicted_error
        return compensated_position.flatten()

    def compensate_offline_speed_first_trajectory(self, trajectory):
        if self.surrogate_model is None:
            raise RuntimeError("代理模型未加载，无法执行离线补偿。")
        predicted_errors = self.surrogate_model.predict(trajectory)
        compensated_trajectory = trajectory - predicted_errors
        return compensated_trajectory

if __name__ == '__main__':
    print("要运行部署演示，请使用 'python main.py' 并将 config.yaml 中的 mode 设置为 'deploy'。")
    pass