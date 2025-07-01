# 假设你有一个形状为 (5, num_epochs) 的 numpy 数组，每一行是一轮实验的 mAP 曲线
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def training_sem():
    map_results = np.array([
        [0.6, 0.65, 0.68, 0.7, 0.72],
        [0.59, 0.66, 0.67, 0.71, 0.73],
        [0.58, 0.64, 0.69, 0.7, 0.74],
        [0.6, 0.63, 0.66, 0.72, 0.75],
        [0.61, 0.67, 0.7, 0.73, 0.76],
    ])


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

def plot_comparison():
    
    # 读取第一个文件数据
    with open('sample/only-cloud.txt', 'r') as f:
        data1 = [float(line.strip()) for line in f.readlines()]

    # 读取第二个文件数据
    with open('sample/only-edge.txt', 'r') as f:
        data2 = [float(line.strip()) for line in f.readlines()]

    # 横坐标，假设是0到100的索引
    x = list(np.arange(0, 101, 5))

    # 绘制曲线
    plt.plot(x, data1, label='Ability on cloud', marker='o')
    plt.plot(x, data2, label='Ability on edge', marker='s')
    # plt.plot(x, data2, label='Ability on cloud-edge', marker='*')

    # 只显示 0, 10, 20, ..., 100
    plt.xticks(np.arange(0, 101, 10))

    plt.title('Comparison of Cloud and Edge Abilities')
    plt.xlabel('Epochs')
    plt.ylabel('mAP@0.5')
    plt.legend()
    plt.grid(True)
    plt.savefig('output.png', dpi=300, bbox_inches='tight')  # 保存为高分辨率PNG




if __name__ == "__main__":
    plot_comparison()