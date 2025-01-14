# 0. 多卡并行训练和推理
```python
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 --master_port=25641 train.py
```
```python
CUDA_VISIBLE_DEVI=0 python predict.py
```

# 1.数据集下载
在网盘中下载数据集并解压到工程的根目录下。
/model_data/voc_classes_ori.txt中是数据集中全部的类别名称
/model_data/voc_classes_2.txt是模拟对2类进行基础训练的名称
/model_data/voc_classes_2.txt是模拟对4类进行基础训练的名称
运行voc_annotation.py生成训练所需的2007_train.txt和2007_val.txt文件

# 2.基础网络训练
train.py中有对于分布式训练以及各种超参数设置
train.py中训练前需要调整的地方：
    1.classes_path    = 'model_data/voc_classes_2.txt'调整成需要训练的类别txt
    2.class_names = ['aeroplane', 'bicycle']改成对应的类别名称
    3.UnFreeze_Epoch为训练轮数、Unfreeze_batch_size为batch数、eval_period为每多少轮测试

单卡对应将参数调整成单卡模式，终端中直接运行python；
多卡调成多卡模式，终端中运行CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
训练完成的权重会储存在logs文件夹中

# 3.中间数据存储
1. 修改/model_data/voc_classes.txt中的类别数量
2. 运行voc_annotation.py生成类别对应的标注文件
3. 修改saveh5.py中的target_class_names，使之与voc_classes.txt中的类别保持一致
4. 修改save_backbone_outputs函数的三个输入：dataloader_val、save_val_path1，save_val_path2，save_val_path3和json文件名val_bboxes.json，运行saveh5.py，生成中间数据的验证集
5. 修改save_backbone_outputs函数的三个输入：dataloader_train、save_train_path1，save_train_path2，save_train_path3和json文件名train_bboxes.json，运行saveh5.py，生成中间数据的训练集


# 4.增量学习训练（*需灵汐配合*）
1. 确保classes_path和class_names包含的类别一致
2. 确保weight_path路径包含骨干特征提取网络
3. 运行train_head.py


# 5.SNN转换与map测试
/ANN2SNN/ANN2SNN.py中有相关转换和测试代码
需要对应修改增量种类用到的classes_path、class_names、训练后的权重地址weight_path
直接在终端运行py文件可储存转换前后的网络、输出转换前后的map
最终是将转换后SNN网络在灵汐板卡上推理（*需灵汐配合*）


# 清华后续优化要点：
1.中间数据储存时对原始类别数据随机储存部分数据；
2.中间数据进行encoding与decoding，储存压缩数据，减小数据存储代价。
注：snn推理与增量模型训练不变，不影响灵汐对模型的适配与部署。
