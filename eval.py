import datetime
import os

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt

import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from utils.utils import cvtColor, preprocess_input, resize_image
from utils.utils_bbox import DecodeBox
from utils.utils_map import get_map
from utils.utils import get_classes
from get_the_classes import get_target_classes, load_data_with_specific_classes
from nets.yolo import YoloBody
from utils.add_param import add_parameters, add_ghostnet
from nets.ReGhos_Block import *
import time


class EvalCallback():
    def __init__(self, net, input_shape, class_names, num_classes, val_lines, log_dir, cuda, \
            map_out_path=".temp_map_out", max_boxes=100, confidence=0.05, nms_iou=0.5, letterbox_image=True, MINOVERLAP=0.5, eval_flag=True, period=1):
        super(EvalCallback, self).__init__()
        
        self.net                = net
        self.input_shape        = input_shape
        self.class_names        = class_names
        self.num_classes        = num_classes
        self.val_lines          = val_lines
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.map_out_path       = map_out_path
        self.max_boxes          = max_boxes
        self.confidence         = confidence
        self.nms_iou            = nms_iou
        self.letterbox_image    = letterbox_image
        self.MINOVERLAP         = MINOVERLAP
        self.eval_flag          = eval_flag
        self.period             = period
        
        self.bbox_util          = DecodeBox(self.num_classes, (self.input_shape[0], self.input_shape[1]))
        
        self.maps       = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"), "w", encoding='utf-8') 
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return 

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]

        top_100     = np.argsort(top_conf)[::-1][:self.max_boxes]
        top_boxes   = top_boxes[top_100]
        top_conf    = top_conf[top_100]
        top_label   = top_label[top_100]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
    
    def on_epoch_end(self, epoch, model_eval):
        
        self.net = model_eval
        if not os.path.exists(self.map_out_path):
            os.makedirs(self.map_out_path)
        if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
            os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
        if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
            os.makedirs(os.path.join(self.map_out_path, "detection-results"))
        print("Getting mAP.")
        for annotation_line in tqdm(self.val_lines):
            line        = annotation_line.split()
            image_id    = os.path.basename(line[0]).split('.')[0]
            #------------------------------#
            #   读取图像并转换成RGB图像
            #------------------------------#
            image       = Image.open(line[0])
            #------------------------------#
            #   获得预测框
            #------------------------------#
            gt_boxes    = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
            #------------------------------#
            #   获得预测txt
            #------------------------------#
            self.get_map_txt(image_id, image, self.class_names, self.map_out_path)
            
            #------------------------------#
            #   获得真实框txt
            #------------------------------#
            with open(os.path.join(self.map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                for box in gt_boxes:
                    left, top, right, bottom, obj = box
                    obj_name = self.class_names[obj]
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
                    
        print("Calculate Map.")
        # try:
        #     temp_map = get_coco_map(class_names = self.class_names, path = self.map_out_path)[1]
        # except:
        temp_map = get_map(self.MINOVERLAP, False, score_threhold=0.5, path = self.map_out_path)
        self.maps.append(temp_map)
        self.epoches.append(epoch)

        with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
            f.write(str(temp_map))
            f.write("\n")
        
        plt.figure()
        plt.plot(self.epoches, self.maps, 'red', linewidth = 2, label='train map')

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Map %s'%str(self.MINOVERLAP))
        plt.title('A Map Curve')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
        plt.cla()
        plt.close("all")

        print("Get map done.")
        shutil.rmtree(self.map_out_path)




def estimate_sram(model, input_size=(1, 3, 640, 640), dtype=torch.float32):
    # 参数字节
    param_size = sum(p.numel() for p in model.parameters()) * torch.finfo(dtype).bits / 8
    print(f"param_size is {param_size / (1024 ** 2):.2f} MB")
    # 梯度字节
    grad_size = param_size  # 每个参数都有一个梯度
    
    # 模拟前向传播，计算特征图大小
    feature_size = 0
    def hook(module, input, output):
        nonlocal feature_size
        if isinstance(output, torch.Tensor):
            feature_size += output.numel() * torch.finfo(dtype).bits / 8
        elif isinstance(output, (list, tuple)):
            for o in output:
                if torch.is_tensor(o):
                    feature_size += o.numel() * torch.finfo(dtype).bits / 8

    hooks = []
    for layer in model.modules():
        hooks.append(layer.register_forward_hook(hook))
    
    dummy_input = torch.randn(*input_size)
    model.eval()
    with torch.no_grad():
        model(dummy_input)
    
    for h in hooks:
        h.remove()

    # 优化器状态（假设 Adam，有 m 和 v 两组参数）
    optimizer_state_size = 8 * param_size

    total_sram = param_size + grad_size + feature_size + optimizer_state_size
    return total_sram / (1024 ** 2)  # 转成 MB




def main():
    classes_path    = 'model_data/voc_classes.txt'
    class_names, num_classes = get_classes(classes_path)
    target_classes = get_target_classes(classes_path, class_names)
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'SNN_logs/latent_12800/latent_12800.pth' # 'weights/first10-large.pth' 

    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir         = os.path.join('eval_logs', "loss_" + str(time_str))
    os.makedirs(log_dir, exist_ok=True)



    # 创建模型
    model = YoloBody((640, 640), num_classes, 'l', False)
    # model = add_ghostnet(model)

    pretrained_dict = torch.load(model_path, map_location = device)
    # 加载参数
    model_dict = model.state_dict()
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():  # 是否只加载骨干的参数，可以替换 renamed_dict
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

    # print(f"pretrained_dict keys: {list(pretrained_dict.keys())}")
    # rand_tensor = torch.randn(1, 3, 640, 640)
    # z = model(rand_tensor)
    # torch.save(z, 'compressed_z.pth')
    # exit(0)
    
        
    # 推理——复制参数给 "original_block."
    new_state_dict = pretrained_dict.copy()
    
    # # cal SRAM
    # sram_size = estimate_sram(model)
    # print(f"Estimated SRAM size: {sram_size:.2f} MB")
    
    # 计算模型复杂度
    from ptflops import get_model_complexity_info
    with torch.no_grad():
        macs, params = get_model_complexity_info(model, (3, 640, 640), as_strings=True,
                                                print_per_layer_stat=False, verbose=False)
    print(f"FLOPs: {macs} / Params: {params}")
    # exit(0)

    # for k, v in pretrained_dict.items():
    #     if "original_block." in k:
    #         new_key = k.replace("original_block.", "")
    #         if new_key not in new_state_dict:
    #             new_state_dict[new_key] = v
    
    # for param in model.parameters():
    #     param.requires_grad = False
    # model.backbone = add_ghostnet(model.backbone)
    
    
    # all_params = sum(p.numel() for p in model.parameters())
    # train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"所有参数为 {all_params* 4 / (1024 * 1024):.2f}MB, 其中可训练参数为 {train_params* 4 / (1024 * 1024):.2f}MB")
    
    model.load_state_dict(model_dict)
    # print(f"model is {model}")
    

    start_t = time.time()
    _, val_lines, _, num_val = load_data_with_specific_classes(train_annotation_path, val_annotation_path, target_classes)
    eval_callback   = EvalCallback(model, (640,640), class_names, num_classes, val_lines, log_dir, True, \
                                    eval_flag=True)
    eval_callback.on_epoch_end(1, model.cuda().eval())
    end_t = time.time()
    inference_time = end_t - start_t  # 秒
    print(f"Inference Time: {inference_time:.2f} s")

if __name__ == "__main__":
    main()
    
