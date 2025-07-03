import time
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlShutdown
import numpy as np
from scipy import stats

def get_power():
    nvmlInit()  # 初始化 NVML
    handle = nvmlDeviceGetHandleByIndex(4)  # 读取第 0 块显卡，可调整索引
    data = []
    # 持续监控 10 秒
    sum_power = 0
    for _ in range(60):
        power = nvmlDeviceGetPowerUsage(handle) / 1000  # 单位：毫瓦，转换为瓦
        # print(f"当前显卡功耗：{power} W")
        data.append(power)
        time.sleep(1)  # 每秒读取一次
    
    mean = np.mean(data)
    sem = stats.sem(data)
    print(f"平均值：{mean:.4f}W、标准误差：{sem:.4f}W")

    nvmlShutdown()  # 释放资源

def get_sem():
    data = [115.74, 116.12, 109.23, 107.83, 101.21, 102.56]
    data1 = [2.1, 1.93, 1.47, 0.21, 0.19, 0.2]
    mean = np.mean(data)
    sem = stats.sem(data)
    print(f"平均值：{mean:.4f}W、标准误差：{sem:.4f}W")

if __name__ == "__main__":
    get_sem()