from matplotlib import pyplot as plt
import os
import numpy as np


def loss_plot():
    losses = []
    val_loss = []
    
    log_dir = '/data/xin/Documents/YOLOv8-CL/log1124'

    with open ('epoch_loss.txt', 'r') as file:
        for line in file:
            # 去掉行末的换行符，并将字符串转换为浮点数后添加到列表中
            losses.append(float(line.strip()))

    with open ('epoch_val_loss.txt', 'r') as file:
        for line in file:
            # 去掉行末的换行符，并将字符串转换为浮点数后添加到列表中
            val_loss.append(float(line.strip()))

    iters = range(len(losses))
    print(f"iters is {iters}")
    plt.figure()
    plt.plot(iters, losses, 'red', linewidth = 2, label='train loss')
    plt.plot(iters, val_loss, 'coral', linewidth = 2, label='val loss')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")
    x_major_locator = plt.MultipleLocator(5)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    plt.savefig(os.path.join(log_dir, "epoch_loss.png"))

    plt.cla()
    plt.close("all")

def plot_map():
    maps = []
    log_dir = '/home/lxx/Documents/YOLOv8-CL/log0114'
    with open ('log0114/epoch_map.txt', 'r') as file:
        for line in file:
            # 去掉行末的换行符，并将字符串转换为浮点数后添加到列表中
            maps.append(float(line.strip()))
    iters = range(len(maps))
    plt.figure()
    plt.plot(iters, maps, 'red', linewidth = 2, label='train map')

    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Map %s'%str(0.5))
    plt.title('A Map Curve')
    plt.legend(loc="upper right")

    plt.savefig(os.path.join(log_dir, "epoch_map.png"))
    plt.cla()
    plt.close("all")

    print("Get map done.")

def plot_map_txt():
    
    # 读取原始数据
    file_path = 'epoch_map.txt'
    with open(file_path, 'r') as file:
        original_data = [float(line.strip()) for line in file]

    # 总轮数和偏移量
    total_epochs = 100
    shift = 10

    # 原始 x 轴和新的 x 轴
    x_original = np.linspace(0, 1, total_epochs)
    x_shifted = np.linspace(0, 1, total_epochs)
    x_shifted = (x_shifted + shift / total_epochs) % 1

    # 对数据进行插值
    interpolated_data = np.interp(x_shifted, x_original, original_data)

    # 将结果保存到文件
    output_path = 'epoch_map2.txt'
    with open(output_path, 'w') as file:
        for value in interpolated_data:
            file.write(f"{value}\n")

    print(f"数据已修改并保存到 {output_path}")


if __name__ == '__main__':
    plot_map()