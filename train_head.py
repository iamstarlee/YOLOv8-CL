import h5py
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from tqdm import tqdm
from nets.yolo_training import Loss, ModelEMA
import torch.optim as optim
from utils.utils import get_classes
from nets.yolo import YoloBody
import os
from utils.utils import get_lr
from utils.callbacks import EvalCallback, LossHistory
import datetime
from get_the_classes import get_target_classes, load_data_with_specific_classes


class Dataset_Head(Dataset):
    def __init__(self, h5_file1, h5_file2, h5_file3, json_file, transform=None):

        self.h5_file1 = h5py.File(h5_file1, 'r')
        self.h5_file2 = h5py.File(h5_file2, 'r')
        self.h5_file3 = h5py.File(h5_file3, 'r')

        with open(json_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.image_ids = list(self.annotations.keys())
        self.transform = transform
        

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Get image ID
        # import pdb; pdb.set_trace()
        image_id = self.image_ids[idx]

        feat1 = self.h5_file1[image_id][:]
        feat2 = self.h5_file2[image_id][:]
        feat3 = self.h5_file3[image_id][:]
        feat1 = torch.tensor(feat1, dtype=torch.float32)
        feat2 = torch.tensor(feat2, dtype=torch.float32)
        feat3 = torch.tensor(feat3, dtype=torch.float32)
        

        boxes = self.annotations[image_id][0][-4:] # 最后四个值为坐标值
        labels = self.annotations[image_id][0][1] # 第二个值为类别值
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.transform:
            feat1 = self.transform(feat1)
            feat2 = self.transform(feat2)
            feat3 = self.transform(feat3)

        return feat1, feat2, feat3, boxes, labels

def count_json_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

if __name__ == '__main__':
    device           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs           = 100
    cuda             = True
    batch_size       = 32
    Init_lr          = 1e-2
    Min_lr           = Init_lr * 0.01
    lr_limit_max     = 5e-2
    lr_limit_min     = 5e-4
    momentum         = 0.937
    input_shape      = [640, 640]
    phi              = 'l'
    classes_path     = './model_data/voc_classes.txt'
    class_names, num_classes = get_classes(classes_path)
    pretrained       = False
    save_dir         = 'SNN_logs'
    class_names      = ['aeroplane', 'bicycle', 'bird', 'boat']
    target_classes   = get_target_classes(classes_path, class_names)
    train_annotation_path   = './2007_train.txt'
    val_annotation_path     = './2007_val.txt'
    train_lines, val_lines, num_train, num_val = load_data_with_specific_classes(train_annotation_path, val_annotation_path, target_classes)
    epoch_step       = num_train // batch_size
    epoch_step_val   = num_val // batch_size
    save_period      = 10



    model = YoloBody(input_shape, num_classes=2, phi='l', pretrained=pretrained)

    for param in model.backbone.parameters():
            param.requires_grad = False


    weight_path = os.path.join("./logs/best_epoch_weights.pth")
    model.load_state_dict(torch.load(weight_path, map_location='cuda'))

    # Create dataset and dataloader
    dataset_train = Dataset_Head('targets/train_backbone_outputs1.h5', 
                                 'targets/train_backbone_outputs2.h5',
                                 'targets/train_backbone_outputs3.h5', 
                                 'targets/train_bboxes.json')
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataset_val = Dataset_Head('targets/val_backbone_outputs1.h5', 
                               'targets/val_backbone_outputs2.h5', 
                               'targets/val_backbone_outputs3.h5', 
                               'targets/val_bboxes.json')
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    
    yolo_loss = Loss(model)

    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    eval_callback   = EvalCallback(model, input_shape, class_names, num_classes, val_lines, log_dir, True, \
                                        eval_flag=True, period=5) # period 是多少轮后开始评估模型
    

    pg0, pg1, pg2 = [], [], []
    nbs             = 64  
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)    
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)    
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)   
    Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    optimizer = optim.SGD(pg0, Init_lr_fit, momentum = momentum, nesterov=True)

    model_train = model.train()
    model_train = model_train.cuda()
    model_train.training_head = True
    ema = ModelEMA(model_train)
    

    # 开始训练 Head
    for epoch in range(epochs):
        loss             = 0.0 
        val_loss         = 0.0
        print('Start Train')
        iteration = 0
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{epochs}',postfix=dict,mininterval=0.3)
        
        for feat1, feat2, feat3, bboxes, labels in dataloader_train:
            optimizer.zero_grad()
            if cuda:
                feat1 = feat1.cuda()
                feat2 = feat2.cuda()
                feat3 = feat3.cuda()
                bboxes = bboxes.cuda()
            
            outputs = model_train((feat1, feat2, feat3))

            loss_value = yolo_loss(outputs, bboxes)

            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients
            optimizer.step()
            
            ema.update(model_train)

            loss += loss_value.item()
            
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)
            
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{epochs}',postfix=dict,mininterval=0.3)

        model_eval = ema.ema # Evaluations that use moving averaged parameters
        for feat1, feat2, feat3, bboxes, labels in dataloader_val:
            with torch.no_grad():
                if cuda:
                    feat1 = feat1.cuda()
                    feat2 = feat2.cuda()
                    feat3 = feat3.cuda()
                    bboxes = bboxes.cuda()
                #----------------------#
                #   清零梯度
                #----------------------#
                optimizer.zero_grad()
                #----------------------#
                #   前向传播
                #----------------------#
                outputs     = model_eval((feat1, feat2, feat3))
                loss_value  = yolo_loss(outputs, bboxes)

            val_loss += loss_value.item()
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)
            iteration = iteration + 1

        # 保存权重
        pbar.close()
        print('Finish Validation')
        model_eval.training_head = False # 下面计算mAP需要走YoloBody forward的另一个分支
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_eval)
        save_state_dict = ema.ema.state_dict()
        model_eval.training_head = True

        if (epoch + 1) % save_period == 0 or epoch + 1 == epochs:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
        
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))