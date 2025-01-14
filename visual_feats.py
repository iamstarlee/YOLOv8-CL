import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from nets.yolo import YoloBody
import torch.nn.functional as F
import numpy as np
from utils.utils import (cvtColor, get_classes, preprocess_input,
                         resize_image, show_config)

def draw_backbone_feature_maps():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = YoloBody(input_shape=[640,640], num_classes=4, phi='l', pretrained=False)
    pretrained_dict = torch.load('SNN_logs/loss_2025_01_11_16_36_59/best_epoch_weights.pth', map_location=device)
    rename_dict = {
                    key.replace("decode_model2.", "") 
                    if key.startswith("decode_model2.") else key: value
                    for key, value in pretrained_dict.items()
                    }
    model.load_state_dict(rename_dict)
    model.to(device)
    model.eval()  # 设置为评估模式

    # 定位 backbone 部分
    backbone = model.backbone if hasattr(model, 'backbone') else model  # 假设模型有 `backbone` 属性


    # 创建示例输入图像
    image = Image.open("img/street.jpg")
    image_shape = np.array(np.shape(image)[0:2])
    image       = cvtColor(image)
    image_data  = resize_image(image, (640, 640), True)
    image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
    images = torch.from_numpy(image_data).to(device)


    # 获取 backbone 输出特征图
    with torch.no_grad():
        feature_map = backbone(images)


    # 处理特征图为灰度图
    print(f"feature_map[0] is {feature_map[2].shape}")
    feat1 = F.interpolate(feature_map[2], size=(40, 40), mode='bilinear', align_corners=False) # 1*256*80*80 => 1*256*40*40
    print(f"feat is {feat1.shape}")
    feature_map = feat1.squeeze(0)  # 去掉第一个通道的 batch 维度

    processed = []
    for i in range(feature_map.shape[0]):  # 遍历每个通道
        processed.append(feature_map[i].data.cpu().numpy())

    

    # 可视化特征图
    fig = plt.figure(figsize=(30, 50))
    for i, fm in enumerate(processed):
        if i == 0:
            a = fig.add_subplot(1, 1, i + 1)  # 假设最多 64 个通道，8x8 网格
            img_plot = plt.imshow(fm, cmap='viridis')
            a.axis("off")
            a.set_title(f"Channel {i+1}", fontsize=12)

    plt.savefig('backbone_feature_maps_3.jpg', bbox_inches='tight')
    print("Backbone feature maps saved as 'backbone_feature_maps.jpg'")


if __name__ == '__main__':
    draw_backbone_feature_maps()
