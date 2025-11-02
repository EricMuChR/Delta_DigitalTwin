import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from meta_learning.maml import MAMLRegressor

def generate_compensation_model(robot, identified_params, settings):
    print("开始生成标准MLP补偿模型...")
    
    ws = settings['workspace_limits']
    grid_density = settings['generalization_grid_density']
    x = np.linspace(ws, ws[1], grid_density)
    y = np.linspace(ws[1], ws, grid_density[1])
    z = np.linspace(ws, ws, grid_density[1])
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z)
    cmd_pos_grid = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
    
    print(f"步骤 1/3: 生成 {len(cmd_pos_grid)} 个虚拟数据点。")

    thetas_nominal = np.array([robot.inverse_kinematics(p) for p in cmd_pos_grid])
    actual_pos_grid = np.array(robot.forward_kinematics_with_errors(thetas_nominal, identified_params))
    error_vectors = actual_pos_grid - cmd_pos_grid
    print("步骤 2/3: 仿真计算真实误差完成。")

    print("步骤 3/3: 训练MLP代理模型 (Scikit-learn内置进度)...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=settings['mlp']['hidden_layers'],
            activation='relu', solver='adam', max_iter=settings['mlp']['max_iter'],
            random_state=42, verbose=True, early_stopping=True, n_iter_no_change=20
        ))
    ])
    pipeline.fit(cmd_pos_grid, error_vectors)
    
    print("MLP模型训练完成。")
    return pipeline

def generate_meta_compensation_model(robot, identified_params, config):
    print("开始生成元学习(MAML)补偿模型...")
    
    maml_settings = config['stage_two_settings']['meta_learning']
    model = MAMLRegressor(settings=maml_settings)
    
    print("正在生成元学习任务...")
    tasks = []
    for _ in tqdm(range(maml_settings['n_tasks']), desc="生成元学习任务"):
        task_specific_params = identified_params + np.random.normal(0, 0.0001, size=identified_params.shape)
        
        n_samples = maml_settings['n_samples_per_task'] * 2
        ws = config['stage_two_settings']['workspace_limits']
        task_cmd_pos = np.random.rand(n_samples, 3)
        task_cmd_pos[:, 0] = task_cmd_pos[:, 0] * (ws[1] - ws) + ws
        task_cmd_pos[:, 1] = task_cmd_pos[:, 1] * (ws - ws[1]) + ws[1]
        task_cmd_pos[:, 2] = task_cmd_pos[:, 2] * (ws - ws) + ws
        
        thetas = np.array([robot.inverse_kinematics(p) for p in task_cmd_pos])
        actual_pos = np.array(robot.forward_kinematics_with_errors(thetas, task_specific_params))
        errors = actual_pos - task_cmd_pos
        
        support_x, query_x = task_cmd_pos[:n_samples//2], task_cmd_pos[n_samples//2:]
        support_y, query_y = errors[:n_samples//2], errors[n_samples//2:]
        
        tasks.append((support_x, support_y, query_x, query_y))

    model.fit(tasks, epochs=maml_settings['meta_epochs'])
    
    print("元学习模型训练完成。")
    return model