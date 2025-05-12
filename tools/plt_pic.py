# 假设你有一个形状为 (5, num_epochs) 的 numpy 数组，每一行是一轮实验的 mAP 曲线
import numpy as np

map_results = np.array([
    [0.6, 0.65, 0.68, 0.7, 0.72],
    [0.59, 0.66, 0.67, 0.71, 0.73],
    [0.58, 0.64, 0.69, 0.7, 0.74],
    [0.6, 0.63, 0.66, 0.72, 0.75],
    [0.61, 0.67, 0.7, 0.73, 0.76],
])
import matplotlib.pyplot as plt
import numpy as np

# 平均值与标准差
mean_map = np.mean(map_results, axis=0)
std_map = np.std(map_results, axis=0)
print(f"mean_map: {mean_map}")
print(f"std_map: {std_map}")
# X 轴为 epoch
epochs = np.arange(1, map_results.shape[1] + 1)

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(epochs, mean_map, label='Mean mAP', color='blue')
plt.fill_between(epochs, mean_map - std_map, mean_map + std_map,
                 color='blue', alpha=0.3, label='±1 std dev')

plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.title('Mean mAP with Standard Deviation')
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig('tools/saved.png')