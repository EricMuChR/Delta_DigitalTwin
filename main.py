import argparse
import yaml
import os
import numpy as np
import pickle
import shutil

from kinematics.delta_robot import DeltaRobot
from calibration.stage_one import identify_parameters
from calibration.stage_two import generate_compensation_model, generate_meta_compensation_model
from utils.data_loader import load_measurement_data
from utils.visualizer import plot_error_comparison
from simulation.closed_loop_validation import run_simulation_validation
from hardware.deploy import RealTimeCompensator
from hardware.map_generator import generate_and_export_error_map

def export_deployment_package(config):
    pkg_settings = config['deployment_settings']['package_export_settings']
    paths = config['paths']
    export_dir = pkg_settings['export_directory']
    model_export_name = pkg_settings['model_filename']
    print(f"\n--- 开始导出一键式部署包至 '{export_dir}' ---")
    if os.path.exists(export_dir): shutil.rmtree(export_dir)
    os.makedirs(export_dir)
    source_model_path = paths['compensation_model_output']
    if config['stage_two_settings']['generalization_mode'] == 'meta': source_model_path = paths['meta_model_output']
    if not os.path.exists(source_model_path): raise FileNotFoundError(f"模型文件 {source_model_path} 不存在。")
    target_model_path = os.path.join(export_dir, model_export_name)
    shutil.copy(source_model_path, target_model_path)
    print(f"  [1/4] 模型文件已复制。")
    api_template_path = "templates/compensator_api_template.py"
    target_api_path = os.path.join(export_dir, "compensator_api.py")
    shutil.copy(api_template_path, target_api_path)
    print(f"  [2/4] API文件已生成。")
    with open(os.path.join(export_dir, "requirements.txt"), 'w') as f: f.write("scikit-learn\nnumpy\n")
    print("  [3/4] 'requirements.txt' 已生成。")
    readme_content = "# Robot Compensation Module\n\n..." # 省略内容
    with open(os.path.join(export_dir, "README.md"), 'w') as f: f.write(readme_content)
    print("  [4/4] 'README.md' 已生成。")
    print(f"\n--- 部署包导出成功！---")

def specialize_and_map_workflow(config):
    """
    新功能：一键式完成模型专机化和误差地图生成。
    """
    print("\n--- 开始执行'专机化并生成地图'工作流 ---")
    settings = config['specialize_and_map_settings']
    
    # 1. 加载预训练的 MAML 模型
    model_path = settings['source_meta_model_path']
    print(f"  [1/4] 正在从 {model_path} 加载MAML元模型...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MAML模型 {model_path} 未找到。请先运行 'generalize' 模式并设置 'meta'。")
    with open(model_path, 'rb') as f:
        maml_model = pickle.load(f)
    print("        元模型加载成功。")

    # 2. 加载新机器人的少量适应数据
    data_path = settings['adaptation_data_path']
    print(f"  [2/4] 正在从 {data_path} 加载适应数据...")
    cmd_pos, meas_pos = load_measurement_data(data_path)
    error_vectors = meas_pos - cmd_pos
    print(f"        加载了 {len(cmd_pos)} 个适应数据点。")

    # 3. 执行模型适应 (专机化)
    print("  [3/4] 正在使用新数据对模型进行快速适应...")
    adapted_params = maml_model.adapt(cmd_pos, error_vectors)
    print("        模型专机化完成。")

    # 创建一个使用已适应参数的预测函数 (作为专机化模型)
    specialized_model = lambda x: maml_model.predict(x, adapted_params=adapted_params)

    # 4. 调用地图生成器
    print("  [4/4] 正在使用专机化模型生成误差地图...")
    temp_config = config.copy()
    temp_config['deployment_settings']['map_export_settings'] = {
        'export_directory': settings['export_directory'],
        'map_filename': settings['map_filename'],
        'map_grid_density': settings['map_grid_density']
    }
    generate_and_export_error_map(specialized_model, temp_config)

def main():
    parser = argparse.ArgumentParser(description="配置驱动的机器人标定程序")
    parser.add_argument('--config', type=str, default='config.yaml', help="主配置文件路径")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"成功加载配置文件: {args.config}")

    # 获取设备配置
    device = config['run_settings'].get('device', 'cpu')
    print(f"\n{'='*60}")
    print(f"全局设备设置: {device.upper()}")
    print(f"{'='*60}")
    
    # 诊断信息：检查 JAX 安装情况
    import jax
    print(f"\nJAX 版本: {jax.__version__}")
    print(f"默认后端: {jax.default_backend()}")
    try:
        print(f"所有可用设备: {jax.devices()}")
        print(f"GPU 设备: {jax.devices('gpu')}")
    except:
        print("未检测到 GPU 设备或 JAX GPU 支持未安装")
    print(f"{'='*60}\n")

    os.makedirs("results", exist_ok=True)

    robot = DeltaRobot(config['robot_parameters'])
    print("机器人模型初始化完成。")

    mode = config['run_settings']['mode']
    paths = config['paths']
    print(f"\n--- 当前运行模式: {mode.upper()} ---")

    if mode == 'simulate':
        run_simulation_validation(robot, config)

    elif mode == 'identify':
        cmd_pos, meas_pos = load_measurement_data(paths['identification_data'])
        identified_params = identify_parameters(robot, cmd_pos, meas_pos, 
                                               config['stage_one_settings'], device=device)
        np.save(paths['identified_params_output'], identified_params)
        print(f"参数辨识完成，结果已保存至: {paths['identified_params_output']}")

    elif mode == 'generalize':
        identified_params = np.load(paths['identified_params_output'])
        gen_mode = config['stage_two_settings']['generalization_mode']
        if gen_mode == 'meta':
            compensation_model = generate_meta_compensation_model(robot, identified_params, 
                                                                 config, device=device)
            output_path = paths['meta_model_output']
        else:
            compensation_model = generate_compensation_model(robot, identified_params, 
                                                           config['stage_two_settings'], device=device)
            output_path = paths['compensation_model_output']
        with open(output_path, 'wb') as f: pickle.dump(compensation_model, f)
        print(f"补偿模型生成完成，已保存至: {output_path}")

    elif mode == 'validate':
        model_path = paths['compensation_model_output']
        if config['stage_two_settings']['generalization_mode'] == 'meta': model_path = paths['meta_model_output']
        cmd_pos, meas_pos = load_measurement_data(paths['validation_data'])
        with open(model_path, 'rb') as f: compensation_model = pickle.load(f)
        error_before = meas_pos - cmd_pos
        predicted_error = compensation_model.predict(cmd_pos)
        compensated_pos = cmd_pos + predicted_error
        error_after = meas_pos - compensated_pos
        abs_error_before = np.linalg.norm(error_before, axis=1)
        abs_error_after = np.linalg.norm(error_after, axis=1)
        print(f"补偿前平均绝对误差: {np.mean(abs_error_before) * 1000:.4f} mm")
        print(f"补偿后平均绝对误差: {np.mean(abs_error_after) * 1000:.4f} mm")
        plot_error_comparison(abs_error_before, abs_error_after, paths['validation_plot_output'])

    elif mode == 'deploy':
        strategy = config['deployment_settings']['strategy']
        
        if strategy == 'package_export':
            export_deployment_package(config)
        elif strategy == 'map_export':
            model_path = paths['compensation_model_output']
            print(f"正在使用已保存的模型 {model_path} 生成地图...")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            generate_and_export_error_map(model, config)
        else:
            model_path = paths['compensation_model_output']
            robot.identified_params = np.load(paths['identified_params_output'])
            compensator = RealTimeCompensator(robot, model_path)
            target_pos = np.array([0.1, 0.2, -0.8])
            trajectory = np.array([[0.1, 0.2, -0.8], [0.11, 0.2, -0.8], [0.12, 0.2, -0.8]])
            if strategy == 'online_accuracy':
                print("\n--- 演示策略一: 在线精度优先 ---")
                comp_online = compensator.compensate_online_accuracy_first(target_pos)
                print(f"原始指令: {target_pos}\n补偿指令: {comp_online}")
            elif strategy == 'offline_pointwise':
                print("\n--- 演示策略二, 方案A: 离线速度优先 (逐点) ---")
                comp_offline_A = compensator.compensate_offline_speed_first_pointwise(target_pos)
                print(f"原始指令: {target_pos}\n补偿指令: {comp_offline_A}")
            elif strategy == 'offline_trajectory':
                print("\n--- 演示策略二, 方案B: 离线速度优先 (轨迹) ---")
                comp_offline_B = compensator.compensate_offline_speed_first_trajectory(trajectory)
                print(f"原始轨迹:\n{trajectory}\n补偿轨迹:\n{comp_offline_B}")
            
    elif mode == 'specialize_and_map':
        specialize_and_map_workflow(config)
    
    print("\n--- 程序执行完毕 ---")

if __name__ == '__main__':
    main()