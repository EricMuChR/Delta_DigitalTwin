import numpy as np
import pickle
import os
import shutil
from tqdm import tqdm

def generate_and_export_error_map(model, config):
    """
    为嵌入式系统生成并导出一个误差地图文件。
    现在接收一个模型对象作为输入。
    """
    map_settings = config['deployment_settings']['map_export_settings']
    ws_limits = config['stage_two_settings']['workspace_limits']
    
    export_dir = map_settings['export_directory']
    map_filename = map_settings['map_filename']
    
    print(f"\n--- 开始生成并导出误差地图至 '{export_dir}' ---")

    # 1. 创建导出目录
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir)

    # 2. 生成密集的网格点
    grid_density = map_settings['map_grid_density']
    x = np.linspace(ws_limits, ws_limits[1], grid_density)
    y = np.linspace(ws_limits, ws_limits, grid_density[1])
    z = np.linspace(ws_limits, ws_limits, grid_density)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
    print(f"  [1/4] 已生成 {len(grid_points)} 个地图网格点。")

    # 3. 使用传入的模型预测所有点的误差
    print(f"  [2/4] 正在计算所有网格点的误差向量...")
    # 为了处理大型地图，分批次进行预测
    batch_size = 8192
    error_vectors = np.zeros_like(grid_points)
    for i in tqdm(range(0, len(grid_points), batch_size), desc="生成地图"):
        batch_points = grid_points[i:i+batch_size]
        error_vectors[i:i+batch_size] = model(batch_points) if callable(model) else model.predict(batch_points)
    print("        误差计算完成。")

    # 4. 保存地图文件和元数据
    map_filepath = os.path.join(export_dir, map_filename)
    metadata = {
        'workspace_limits': ws_limits,
        'grid_density': grid_density
    }
    np.savez_compressed(
        map_filepath,
        grid_points=grid_points,
        error_vectors=error_vectors,
        metadata=metadata
    )
    print(f"  [3/4] 误差地图已保存至: {map_filepath}")

    # 5. 生成使用说明
    readme_content = f"""
# Robot Error Map Package

This package contains a pre-computed error map for robot compensation, designed for embedded systems.

## Contents

- `{map_filename}`: A compressed NumPy file containing the error map data.
- `map_usage_example.py`: A Python script demonstrating how to load and use the map.
- `README.md`: This file.

## How to Use the Error Map

The `{map_filename}` file contains three main objects:
- `grid_points`: An (N, 3) array of coordinates for every point in the map.
- `error_vectors`: An (N, 3) array of the corresponding error vectors to be subtracted from the command.
- `metadata`: A dictionary containing information about the map, such as `workspace_limits` and `grid_density`.

Your embedded application should load this data and implement a **trilinear interpolation** function to find the error for any given target point within the workspace.

See `map_usage_example.py` for a practical demonstration in Python using `scipy`. For C/C++, you will need to implement the lookup and interpolation logic manually.
"""
    example_content = """
import numpy as np
from scipy.interpolate import griddata
import os

def load_and_use_map():
    # --- 1. 加载误差地图 ---
    map_filename = 'error_map.npz'
    if not os.path.exists(map_filename):
        print(f"错误: 地图文件 '{map_filename}' 未找到。")
        return

    print(f"正在加载地图: {map_filename}")
    error_map = np.load(map_filename, allow_pickle=True)
    grid_points = error_map['grid_points']
    error_vectors = error_map['error_vectors']
    metadata = error_map['metadata'].item()
    
    print("地图加载成功！")
    print(f"地图元数据: {metadata}")

    # --- 2. 定义一个或多个目标点 ---
    target_point = np.array([0.1, 0.2, -0.8])
    
    # --- 3. 使用插值计算误差 ---
    # scipy.interpolate.griddata 是一个功能强大的N维插值函数
    # 'linear' 方法对应三维空间中的三线性插值
    print(f"\\n正在为目标点 {target_point} 进行插值计算...")
    interpolated_error = griddata(grid_points, error_vectors, target_point, method='linear')

    # 检查插值是否成功 (如果点在地图外，结果可能为nan)
    if np.isnan(interpolated_error).any():
        print("警告: 目标点可能在地图定义的凸包之外，无法进行线性插值。")
        # 可以选择使用最近邻插值作为备用方案
        interpolated_error = griddata(grid_points, error_vectors, target_point, method='nearest')
        print(f"已切换到最近邻插值，误差为: {interpolated_error}")

    # --- 4. 计算最终补偿指令 ---
    # 补偿指令 = 原始指令 - 预测误差
    compensated_command = target_point - interpolated_error

    print(f"\\n--- 结果 ---")
    print(f"目标点: {target_point}")
    print(f"插值得到的误差: {interpolated_error}")
    print(f"最终补偿指令: {compensated_command}")

if __name__ == '__main__':
    load_and_use_map()
"""
    with open(os.path.join(export_dir, "README.md"), 'w') as f:
        f.write(readme_content)
    with open(os.path.join(export_dir, "map_usage_example.py"), 'w') as f:
        f.write(example_content)
    
    print(f"  [4/4] 使用说明和示例代码已生成。")
    print(f"\n--- 误差地图导出成功！---")