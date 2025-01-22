import h5py
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils.dataloader import feats_dataset_collate
from torchvision.transforms import functional as F
from tqdm import tqdm
from nets.yolo_training import Loss, ModelEMA
import torch.optim as optim
from utils.utils import get_classes
from nets.yolo import YoloBody
from nets.yolo_training import set_optimizer_lr, get_lr_scheduler
import os
from utils.utils import get_lr
from utils.callbacks import EvalCallback, LossHistory
import datetime
from get_the_classes import get_target_classes, load_data_with_specific_classes
import numpy as np
from functools import partial
from utils.utils import worker_init_fn
import time


class Dataset_Head(Dataset):
    def __init__(self, h5_file1, json_file, transform=None):

        self.h5_file1 = h5py.File(h5_file1, 'r')

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

        feat1 = self.h5_file1[image_id][:] # 从h5文件中读取特征
        feat1 = np.array(feat1, dtype=np.float32)


        # 需要将boxes转换为YoloDataset中的格式
        len_ann = len(self.annotations[image_id])
        labels_out = np.zeros((len_ann, 6))
        for i in range(len_ann):
            boxes = self.annotations[image_id][i][-4:] # 最后四个值为坐标值
            labels = self.annotations[image_id][i][1] # 第二个值为类别值
            # boxes = torch.tensor(boxes, dtype=torch.float32)
            # labels = torch.tensor(labels, dtype=torch.int64)
            boxes  = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            labels_out[i][1] = labels
            labels_out[i][2:] = boxes

        if self.transform:
            feat1 = self.transform(feat1)

        return feat1, labels_out

def count_json_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

if __name__ == '__main__':
    device           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs           = 50
    cuda             = True
    batch_size       = 1
    Init_lr          = 1e-2
    Min_lr           = Init_lr * 0.01
    lr_limit_max     = 5e-2
    lr_limit_min     = 5e-4
    momentum         = 0.937
    weight_decay     = 5e-4
    input_shape      = [640, 640]
    phi              = 'l'
    classes_path     = './model_data/voc_classes.txt'
    class_names, num_classes = get_classes(classes_path)
    pretrained       = False
    save_dir         = 'SNN_logs'
    weight_path      = os.path.join('SNN_logs/loss_2025_01_17_17_44_28/best_epoch_weights.pth')
    lr_decay_type    = "cos"
    class_names      = ['aeroplane', 'bicycle', 'bird', 'boat']
    target_classes   = get_target_classes(classes_path, class_names)
    train_annotation_path   = './2007_train.txt'
    val_annotation_path     = './2007_val.txt'
    train_lines, val_lines, num_train, num_val = load_data_with_specific_classes(train_annotation_path, val_annotation_path, target_classes)
    epoch_step       = num_train // batch_size
    epoch_step_val   = num_val // batch_size
    save_period      = 10


    # Here, num_classes = 4
    model = YoloBody(input_shape, num_classes=num_classes, phi='l', pretrained=pretrained)

    # Load backbone param
    model_dict      = model.state_dict()
    
    pretrained_dict = torch.load(weight_path, map_location='cuda')
    backbone_weights = {k: v for k, v in pretrained_dict.items() if k.startswith("backbone")}
    except_dfl = {k : v for k, v in pretrained_dict.items() if 'dfl' not in k}
    
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in backbone_weights.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")


    # # Freeze backbone
    # for param in model.backbone.parameters():
    #     param.requires_grad = False
    # # Freeze backbone and vae
    for name, param in model.named_parameters():
        if 'encoder' not in name or 'fc_mu' not in name or 'fc_var' not in name or 'decoder_input' not in name or 'decoder' not in name or 'final_layer' not in name:
            param.requires_grad = False
    
    # Create dataset and dataloader
    dataset_train = Dataset_Head('targets/train_backbone_outputs.h5', 
                                 'targets/train_bboxes.json')
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True,
                                drop_last=True, collate_fn=feats_dataset_collate, sampler=None, 
                                worker_init_fn=partial(worker_init_fn, rank=0, seed=42))
    dataset_val = Dataset_Head('targets/val_backbone_outputs.h5', 
                               'targets/val_bboxes.json')
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, pin_memory=True, 
                                drop_last=True, collate_fn=feats_dataset_collate, sampler=None, 
                                worker_init_fn=partial(worker_init_fn, rank=0, seed=42))
    
    yolo_loss = Loss(model)

    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    eval_callback   = EvalCallback(model, input_shape, class_names, num_classes, val_lines, log_dir, True, \
                                        eval_flag=True, period=2) # period 是多少轮后开始评估模型
    

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
    Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    optimizer = optim.SGD(pg0, Init_lr_fit, momentum = momentum, nesterov=True)
    # 获得学习率下降的公式
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, epochs)

    # add_param_group() 用来冻结参数
    optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
    optimizer.add_param_group({"params": pg2})

    model_train = model.train()
    model_train = model_train.cuda()
    model_train.is_vae = True
    ema = ModelEMA(model_train)
    

    # Here we go!
    for epoch in range(epochs):
        loss             = 0.0 
        val_loss         = 0.0
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        print('Start Train')
        
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{epochs}',postfix=dict,mininterval=0.3)
        
        for iteration, (feat1, feat2, feat3, labels) in enumerate(dataloader_train):
            start_time = time.time()
            optimizer.zero_grad()
            # if cuda:
            feat1 = feat1.cuda()
            feat2 = feat2.cuda()
            feat3 = feat3.cuda()
            labels = labels.cuda()
            
            end_time1 = time.time()
            # print(f"load data costs {end_time1 - start_time}")
            outputs = model_train((feat1, feat2, feat3))
            end_time2 = time.time()
            # print(f"train data costs {end_time2 - end_time1}")

            loss_value = yolo_loss(outputs, labels)

            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients
            optimizer.step()
            
            end_time3 = time.time()
            # print(f"back prop costs {end_time3 - end_time2}")
            # ema.update(model_train)

            loss += loss_value.item()
            
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 'lr'    : get_lr(optimizer)})
            pbar.update(1)
            end_time4 = time.time()
            # print(f"end of al costs {end_time4 - end_time3}")
            
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{epochs}',postfix=dict,mininterval=0.3)

        # model_eval = ema.ema # Evaluations that use moving averaged parameters
        model_eval = model_train.eval()
        for iteration, (feat1, feat2, feat3, labels) in enumerate(dataloader_val):
            with torch.no_grad():
                # if cuda:
                feat1 = feat1.cuda()
                feat2 = feat2.cuda()
                feat3 = feat3.cuda()
                labels = labels.cuda()
                #----------------------#
                #   清零梯度
                #----------------------#
                optimizer.zero_grad()
                #----------------------#
                #   前向传播
                #----------------------#
                outputs     = model_eval((feat1, feat2, feat3))
                loss_value  = yolo_loss(outputs, labels)

            val_loss += loss_value.item()
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

        # 保存权重
        pbar.close()
        print('Finish Validation')
        model_eval.training_head = False # 下面计算mAP需要走YoloBody forward的另一个分支
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_eval)
        # save_state_dict = ema.ema.state_dict()
        save_state_dict = model.state_dict()
        model_eval.training_head = True

        if (epoch + 1) % save_period == 0 or epoch + 1 == epochs:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
        
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))