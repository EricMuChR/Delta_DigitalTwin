import numpy as np
from calibration.stage_one import identify_parameters

def run_simulation_validation(robot, config):
    sim_settings = config['simulation_settings']
    paths = config['paths']
    num_params = 42

    print("--- 步骤A: 设定虚拟真实误差参数 ---")
    if sim_settings.get('ground_truth_params') and sim_settings['ground_truth_params']:
        ground_truth_params = np.array(sim_settings['ground_truth_params'])
    else:
        low = config['stage_one_settings']['global_search']['param_lower_bound']
        high = config['stage_one_settings']['global_search']['param_upper_bound']
        ground_truth_params = np.random.uniform(low, high, num_params)
    np.save(paths['sim_ground_truth_params'], ground_truth_params)
    print(f"  '真实' 误差参数已生成并保存至 {paths['sim_ground_truth_params']}。")

    print("\n--- 步骤B: 生成仿真测量数据 ---")
    ws = sim_settings['workspace_limits']
    n_points = sim_settings['num_points']
    cmd_pos = np.random.rand(n_points, 3)
    cmd_pos[:, 0] = cmd_pos[:, 0] * (ws[1] - ws[0]) + ws[0]
    cmd_pos[:, 1] = cmd_pos[:, 1] * (ws[3] - ws[2]) + ws[2]
    cmd_pos[:, 2] = cmd_pos[:, 2] * (ws[5] - ws[4]) + ws[4]
    
    thetas = np.array([robot.inverse_kinematics(p) for p in cmd_pos])
    actual_pos_no_noise = np.array([robot.forward_kinematics_with_errors(theta, ground_truth_params) for theta in thetas])
    
    noise = np.random.normal(sim_settings['noise_mean'], sim_settings['noise_std'], actual_pos_no_noise.shape)
    meas_pos = actual_pos_no_noise + noise
    print(f"  已生成 {sim_settings['num_points']} 组带噪声的仿真测量数据。")

    print("\n--- 步骤C: 运行参数辨识流程 ---")
    identified_params = identify_parameters(robot, cmd_pos, meas_pos, config['stage_one_settings'])
    np.save(paths['sim_identified_params'], identified_params)

    print("\n--- 步骤D: 评估验证结果 ---")
    param_diff_norm = np.linalg.norm(identified_params - ground_truth_params)
    param_relative_diff = param_diff_norm / (np.linalg.norm(ground_truth_params) + 1e-9)
    print(f"  - 参数向量欧氏距离: {param_diff_norm:.6f}")
    print(f"  - 参数向量相对误差: {param_relative_diff:.2%}")

    actual_pos_reproduced = np.array([robot.forward_kinematics_with_errors(theta, identified_params) for theta in thetas])
    repro_error = np.mean(np.linalg.norm(actual_pos_reproduced - actual_pos_no_noise, axis=1))
    print(f"  - 辨识模型的位置复现平均误差: {repro_error * 1000:.6f} mm")

    if repro_error < 1e-4:
        print("\n[结论]: 成功！辨识出的模型能高度复现原始误差，算法流程有效。")
    else:
        print("\n[结论]: 警告！辨识模型未能有效复现原始误差，请检查算法参数或模型定义。")