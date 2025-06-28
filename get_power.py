import time
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlShutdown

nvmlInit()  # 初始化 NVML
handle = nvmlDeviceGetHandleByIndex(5)  # 读取第 0 块显卡，可调整索引

# 持续监控 10 秒
for _ in range(10):
    power = nvmlDeviceGetPowerUsage(handle) / 1000  # 单位：毫瓦，转换为瓦
    print(f"当前显卡功耗：{power} W")
    time.sleep(1)  # 每秒读取一次

nvmlShutdown()  # 释放资源
