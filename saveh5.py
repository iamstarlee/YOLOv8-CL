import h5py
import os
import numpy as np
import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from nets.yolo import YoloBody
from utils.utils import get_classes
from utils.dataloader import YoloDataset, yolo_dataset_collate
import torch
import torch.nn as nn


def save_backbone_outputs(model, dataloader, class_names, target_class_names, input_shape, save_path1, save_path2, save_path3, target_dir):
    model.eval()  # 设置模型为评估模式
    device = next(model.parameters()).device  # 获取模型设备
    target_class_indices = [class_names.index(cls_name) for cls_name in target_class_names]

    # 创建 HDF5 文件和 JSON 文件
    h5_file1 = h5py.File(save_path1, 'w')
    h5_file2 = h5py.File(save_path2, 'w')
    h5_file3 = h5py.File(save_path3, 'w')
    bbox_data = {}

    with torch.no_grad():
        for iteration, batch in enumerate(tqdm(dataloader, desc="Saving intermediate results")):
            images, bboxes = batch  # 解包 batch
            images = images.to(device)
            
            # 处理当前图像的所有边界框
            img_bboxes = bboxes  # 当前批次的边界框
            contains_target_class = False

            # 检查当前图像的 bboxes 是否包含目标类别
            for bbox in img_bboxes:  # 遍历当前图像的所有边界框
                bbox = bbox.tolist()  # 将 tensor 转换为 list
                if len(bbox) == 6:  # 确保 bbox 有正确的元素数量
                    _, class_id, _, _, _, _ = bbox  # 只提取类别 ID
                    if class_id in target_class_indices:  # 检查类别是否属于目标类别
                        contains_target_class = True
                        
                        break  # 找到目标类别后可以跳出循环

            # 如果当前图像包含目标类别的边界框，则保存特征和所有边界框
            if contains_target_class:
                features = model(images)  # 提取特征，features 是一个包含多个张量的 tuple
                
                # 保存特征
                img_feature1 = features[0].squeeze(0).cpu().numpy()
                img_feature2 = features[1].squeeze(0).cpu().numpy()
                img_feature3 = features[2].squeeze(0).cpu().numpy()
                img_id = f'image_{iteration}'  # 生成唯一的图像 ID
                
                h5_file1.create_dataset(img_id, data=img_feature1)
                h5_file2.create_dataset(img_id, data=img_feature2)
                h5_file3.create_dataset(img_id, data=img_feature3)

                # 保存所有边界框。稍作修改，将类别id转为int
                filtered_bboxes = []
                for bbox in img_bboxes:
                    if(len(bbox) == 6):
                        t_bbox = bbox.tolist()
                        t_bbox[1] = int(t_bbox[1])
                        filtered_bboxes.append(t_bbox)

                bbox_id = f'image_{iteration}'  # 生成唯一的图像 ID
                bbox_data[bbox_id] = filtered_bboxes

    h5_file1.close()
    h5_file2.close()
    h5_file3.close()

    # 写入 JSON 文件
    with open(os.path.join(target_dir, 'val_bboxes.json'), 'w') as json_file:
        json.dump(bbox_data, json_file)

    print('特征和边界框已成功保存。')


def get_target_classes(classes_path, class_names):

        with open(classes_path, 'r', encoding='utf-8') as f:
            class_list = [line.strip() for line in f.readlines()]
        target_classes = []
        for name in class_names:
            if name in class_list:
                class_id = class_list.index(name)
                target_classes.append(class_id)
            else:
                print(f"Warning: Class '{name}' not found in {classes_path}")
        
        return target_classes


def load_data_with_specific_classes(train_annotation_path, val_annotation_path, target_classes):

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()

    def filter_lines(lines, target_classes):
        filtered_lines = []
        for line in lines:
            entries = line.strip().split()
            for entry in entries[1:]:
                class_id = int(entry.split(',')[-1])
                if class_id in target_classes:
                    filtered_lines.append(line)
                    break
        return filtered_lines

    train_lines = filter_lines(train_lines, target_classes)
    val_lines = filter_lines(val_lines, target_classes)

    num_train = len(train_lines)
    num_val = len(val_lines)

    return train_lines, val_lines, num_train, num_val


def test(model, dataloader):
    with torch.no_grad():
        for iteration, batch in enumerate(tqdm(dataloader, desc="Saving intermediate results")):
            images, bboxes = batch  # 解包 batch
            images = images.cuda()
            
            # 处理当前图像的所有边界框
            img_bboxes = bboxes  # 当前批次的边界框
            contains_target_class = False
            print(f"bboxes are {bboxes}")
            # 检查当前图像的 bboxes 是否包含目标类别
            for bbox in img_bboxes:  # 遍历当前图像的所有边界框
                
                bbox = bbox.tolist()  # 将 tensor 转换为 list
                if len(bbox) == 6:  # 确保 bbox 有正确的元素数量
                    _, class_id, _, _, _, _ = bbox  # 只提取类别 ID
                    print(f"class_id is {class_id}")


if __name__ == "__main__":
    classes_path = 'model_data/voc_classes.txt'
    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'
    model_path = './logs/best_epoch_weights.pth'
    input_shape = [640, 640]
    batch_size = 1
    phi = 'l'
    pretrained = False

    class_names, num_classes = get_classes(classes_path)
    target_class_names = ['aeroplane', 'bicycle', 'bird', 'boat']  # 指定类别名
    target_dir = 'targets'
    save_train_path1 = os.path.join(target_dir, 'train_backbone_outputs1.h5')
    save_train_path2 = os.path.join(target_dir, 'train_backbone_outputs2.h5')
    save_train_path3 = os.path.join(target_dir, 'train_backbone_outputs3.h5')
    save_val_path1 = os.path.join(target_dir, 'val_backbone_outputs1.h5')
    save_val_path2 = os.path.join(target_dir, 'val_backbone_outputs2.h5')
    save_val_path3 = os.path.join(target_dir, 'val_backbone_outputs3.h5')

    # 创建yolo模型并加载预训练权重
    model = YoloBody(input_shape, num_classes=num_classes, phi='l', pretrained=pretrained)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')), strict=True)
    model = model.cuda()
    Backbone = nn.Sequential(*list(model.children())[:1])

    # 创建数据集
    target_classes = get_target_classes(classes_path, class_names)
    train_lines, val_lines, num_train, num_val = load_data_with_specific_classes(train_annotation_path, val_annotation_path, target_classes)


    mosaic              = True
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7

    dataset_train = YoloDataset(train_lines, input_shape, num_classes, epoch_length=1, \
                                        mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True, special_aug_ratio=special_aug_ratio)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=yolo_dataset_collate)

    dataset_val = YoloDataset(val_lines, input_shape, num_classes, epoch_length=1, \
                                        mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=False, special_aug_ratio=special_aug_ratio)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=yolo_dataset_collate)

    save_backbone_outputs(Backbone, dataloader_val, class_names, target_class_names, input_shape, save_val_path1, save_val_path2, save_val_path3, target_dir)