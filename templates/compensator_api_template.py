import pickle
import numpy as np
import os

class PackagedCompensator:
    """
    一个自包含的误差补偿器API，专为集成到外部项目而设计。
    """
    def __init__(self, model_filename="model.pkl"):
        """
        初始化补偿器。

        Args:
            model_filename (str): 模型文件的名称。默认情况下，它会查找
                                  与此脚本位于同一目录下的 'model.pkl' 文件。
        """
        # 构建模型文件的绝对路径
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, model_filename)
        
        print(f"正在从 {model_path} 加载补偿模型...")
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("模型加载成功，补偿器已就绪。")
        except FileNotFoundError:
            print(f"错误: 模型文件 {model_path} 未找到！")
            raise

    def compensate_point(self, command_position):
        """
        对单个指令点进行误差补偿。

        Args:
            command_position (list, tuple, or np.ndarray): 长度为3的原始指令位置 [x, y, z]。

        Returns:
            np.ndarray: 长度为3的补偿后指令位置。
        """
        command_pos_reshaped = np.array(command_position).reshape(1, -1)
        
        # 使用模型预测该位置的误差
        predicted_error = self.model.predict(command_pos_reshaped)
        
        # 补偿后的位置 = 原始指令 - 预测误差
        compensated_position = command_pos_reshaped - predicted_error
        
        return compensated_position.flatten()

    def compensate_trajectory(self, trajectory):
        """
        对整个轨迹（点集）进行批量误差补偿。

        Args:
            trajectory (np.ndarray): N x 3 的原始轨迹点集。

        Returns:
            np.ndarray: N x 3 的补偿后轨迹点集。
        """
        if not isinstance(trajectory, np.ndarray) or trajectory.ndim != 2 or trajectory.shape[1] != 3:
            raise ValueError("输入轨迹必须是一个 N x 3 的 NumPy 数组。")
            
        predicted_errors = self.model.predict(trajectory)
        compensated_trajectory = trajectory - predicted_errors
        
        return compensated_trajectory

if __name__ == '__main__':
    """
    这是一个演示如何使用本API的示例。
    在您的项目中，您应该 `from compensator_api import PackagedCompensator` 并使用该类。
    """
    print("\n--- PackagedCompensator API 使用演示 ---")
    
    # 1. 初始化补偿器
    # 假设 'model.pkl' 文件和这个脚本在同一个文件夹下
    try:
        compensator = PackagedCompensator()

        # 2. 演示逐点补偿
        print("\n--- 逐点补偿演示 ---")
        target_point = [0.1, 0.2, -0.8]
        compensated_point = compensator.compensate_point(target_point)
        print(f"原始指令: {target_point}")
        print(f"补偿指令: {compensated_point}")

        # 3. 演示轨迹补偿
        print("\n--- 轨迹补偿演示 ---")
        target_trajectory = np.array([[0.1, 0.2, -0.8],
            [0.11, 0.2, -0.8],
            [0.12, 0.2, -0.8],
            [-0.3, -0.3, -0.95]])
        compensated_trajectory = compensator.compensate_trajectory(target_trajectory)
        print("原始轨迹:")
        print(target_trajectory)
        print("\n补偿后的轨迹:")
        print(compensated_trajectory)

    except Exception as e:
        print(f"\n运行演示时出错: {e}")
        print("请确保 'model.pkl' 文件存在于此脚本所在的目录中。")