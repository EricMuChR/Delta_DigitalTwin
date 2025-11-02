import numpy as np

def load_measurement_data(filepath):
    try:
        data = np.genfromtxt(filepath, delimiter=',', skip_header=1, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] != 6:
            raise ValueError(f"数据文件应有6列，但检测到 {data.shape[1]} 列。")
        cmd_pos = data[:, :3]
        meas_pos = data[:, 3:]
        print(f"成功从 {filepath} 加载了 {len(data)} 条数据。")
        return cmd_pos, meas_pos
    except Exception as e:
        print(f"加载数据文件 {filepath} 时出错: {e}")
        raise