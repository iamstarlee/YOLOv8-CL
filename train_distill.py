import datetime
import os

import torch
import matplotlib
matplotlib.use('Agg')
import numpy as np
from utils.utils import get_classes
from get_the_classes import get_target_classes, load_data_with_specific_classes
from nets.yolo import YoloBody

def main():
    classes_path    = 'model_data/voc_classes.txt'
    class_names, num_classes = get_classes(classes_path)
    target_classes = get_target_classes(classes_path, class_names)
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'weights/first10-large.pth'

    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir         = os.path.join('eval_logs', "loss_" + str(time_str))
    os.makedirs(log_dir, exist_ok=True)



    # 创建模型
    tea_model = YoloBody((640, 640), num_classes, 'l', False)
    stu_model = YoloBody((640, 640), num_classes, 'n', False)

    pretrained_dict = torch.load(model_path, map_location = device)

    tea_model.load_state_dict(pretrained_dict)





if __name__ == "__main__":
    main()