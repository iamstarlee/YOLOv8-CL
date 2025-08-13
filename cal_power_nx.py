import subprocess
import time
import re

def monitor_avg_power(duration_sec=10):
    cmd = ["sudo", "tegrastats", "--interval", "1000"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)

    power_values = []
    start_time = time.time()
    
    pattern = re.compile(r'VDD_IN\s+(\d+)mW/(\d+)mW')

    try:
        while time.time() - start_time < duration_sec:
            line = process.stdout.readline()
            match = pattern.search(line)
            if match:
                avg_power = int(match.group(2))  # 提取平均功耗
                power_values.append(avg_power)
    finally:
        process.terminate()

    if power_values:
        print(f"平均功耗：{sum(power_values) / len(power_values):.2f} mW")
    else:
        print("未获取到功耗数据")

# 示例：监控 15 秒
monitor_avg_power(duration_sec=15)
