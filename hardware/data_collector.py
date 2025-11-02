import time
import numpy as np

class RobotController:
    def __init__(self, ip_address):
        self.ip = ip_address
        print(f"正在连接机器人控制器 at {self.ip}...")
        print("机器人已连接。")

    def move_to(self, position, wait_for_completion=True):
        print(f"命令机器人移动到: {position}")
        if wait_for_completion:
            time.sleep(1)
        print("移动完成。")

class MeasurementDevice:
    def __init__(self, device_type='LaserTracker'):
        self.type = device_type
        print(f"正在初始化测量设备: {self.type}...")
        print("测量设备就绪。")

    def get_current_position(self):
        print("正在获取当前测量位置...")
        measured_pos = np.random.rand(3)
        print(f"测量到位置: {measured_pos}")
        return measured_pos

def collect_data_routine(robot, tracker, measurement_points):
    collected_data = []
    for point in measurement_points:
        robot.move_to(point)
        actual_pos = tracker.get_current_position()
        collected_data.append(np.concatenate([point, actual_pos]))
    return np.array(collected_data)

if __name__ == '__main__':
    points_to_measure = np.array([0.1, 0.2, -0.8],
        [-0.2, 0.3, -0.85],
        [0.4, -0.1, -0.9])
    
    my_robot = RobotController(ip_address="192.168.1.10")
    my_tracker = MeasurementDevice()
    
    data = collect_data_routine(my_robot, my_tracker, points_to_measure)
    
    np.savetxt("collected_data.csv", data, delimiter=",", header="cmd_x,cmd_y,cmd_z,meas_x,meas_y,meas_z", comments="")
    print("数据采集完成并已保存。")