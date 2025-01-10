import os
import shutil
import xml.etree.ElementTree as ET
import xml.dom.minidom
import cv2
from collections import Counter
import random
import re

def count_object_names_in_xml(root_folder, output_file):
    # 初始化计数器
    name_counter = Counter()

    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".xml"):
                file_path = os.path.join(root, file)
                
                # 解析XML文件
                tree = ET.parse(file_path)
                root_element = tree.getroot()
                
                # 查找所有<object><name>标签
                for obj in root_element.findall(".//object"):
                    name = obj.find("name")
                    if name is not None:
                        name_counter[name.text] += 1
                

    # 将结果写入txt文件
    with open(output_file, "w") as f:
        for name, count in name_counter.items():
            f.write(f"{name}: {count}\n")
            # f.write(f"{name}\n")


def organize_files(root_folder, annotation_folder, images_folder):
    # 创建输出文件夹，如果不存在则创建
    os.makedirs(annotation_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)

    # 递归遍历所有子文件夹
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".xml"):
                xml_file_path = os.path.join(root, file)
                
                # 构造对应的 jpg 文件名
                base_name = os.path.splitext(file)[0]
                jpg_file_path_lower = os.path.join(root, base_name + ".jpg")
                jpg_file_path_upper = os.path.join(root, base_name + ".JPG")
                
                # 检查是否存在对应的 jpg 文件（无论是小写还是大写）
                if os.path.exists(jpg_file_path_lower):
                    jpg_file_path = jpg_file_path_lower
                elif os.path.exists(jpg_file_path_upper):
                    jpg_file_path = jpg_file_path_upper
                else:
                    # 如果没有找到对应的图片文件，则跳过此文件
                    continue
                
                # 将 XML 文件复制到 Annotation 文件夹
                shutil.copy2(xml_file_path, os.path.join(annotation_folder, file))
                
                # 将图片文件复制到 Images 文件夹
                shutil.copy2(jpg_file_path, os.path.join(images_folder, os.path.basename(jpg_file_path)))

    print("文件已成功整理到指定文件夹中。")





def split_dataset(root_folder, train_ratio=0.7, val_ratio=0.2):
    # 设置文件夹路径
    annotations_folder = os.path.join(root_folder, "Annotations")
    imagesets_folder = os.path.join(root_folder, "ImageSets", "Main")
    
    # 获取所有标注文件的名称（去掉 .xml 后缀）
    all_files = [f.split(".")[0] for f in os.listdir(annotations_folder) if f.endswith(".xml")]
    
    # 随机打乱文件顺序
    random.shuffle(all_files)
    
    # 按比例划分数据集
    total_count = len(all_files)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    
    train_files = all_files[:train_count]
    val_files = all_files[train_count:train_count + val_count]
    test_files = all_files[train_count + val_count:]
    trainval_files = train_files + val_files

    # 定义文件路径
    file_paths = {
        "train.txt": train_files,
        "val.txt": val_files,
        "test.txt": test_files,
        "trainval.txt": trainval_files
    }

    # 将文件列表写入对应的txt文件中
    for filename, file_list in file_paths.items():
        with open(os.path.join(imagesets_folder, filename), "w") as f:
            for item in file_list:
                f.write(f"{item}\n")
    
    print("数据集划分和索引文件已生成。")



def count_files(folder_path):
    # ls -l|grep "^-"| wc -l
    file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    return file_count





def clean_filename(filename):
    # 替换中文符号为空格或相应的英文符号
    filename = filename.replace("，", ",").replace("。", ".").replace("！", "!").replace("？", "?")
    filename = filename.replace("（", "(").replace("）", ")").replace("【", "[").replace("】", "]")
    filename = filename.replace("：", ":").replace("；", ";")
    
    # 去掉空格
    filename = filename.replace(" ", "")
    
    # 移除其他非英文和非数字的符号
    filename = re.sub(r'[^\w\-.()]', '', filename)

    return filename

def rename_files_in_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            # 获取新文件名
            new_name = clean_filename(file)
            
            # 如果新文件名和旧文件名不同，则重命名文件
            if new_name != file:
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, new_name)
                
                # 重命名文件
                os.rename(old_path, new_path)
                print(f"重命名: {file} -> {new_name}")




def clean_xml(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".xml"):
                xml_path = os.path.join(root, file)
                
                # 解析XML文件
                tree = ET.parse(xml_path)
                root_element = tree.getroot()
                
                # 查找<filename>标签
                filename_element = root_element.find("filename")
                if(filename_element is not None and filename_element.text):
                    
                    # # rename xml
                    # # 分离文件名主体和扩展名
                    # base_name, _ = os.path.splitext(filename_element.text)
                    # # 将扩展名改为 .jpg
                    # new_filename = base_name + ".jpg"
                    # # 更新 <filename> 标签内容
                    # if filename_element.text != new_filename:
                    #     filename_element.text = new_filename
                    #     tree.write(xml_path, encoding="utf-8")
                    #     print(f"更新: {xml_path} - filename 修改为: {new_filename}")

                    # clean names of xml
                    origin_text = filename_element.text
                    cleaned_text = clean_filename(origin_text)
                
                    if cleaned_text != origin_text:
                        filename_element.text = cleaned_text
                        tree.write(xml_path, encoding = 'utf-8')
                        print(f"文件已修改: {xml_path} - 原始： {origin_text}->新：{cleaned_text}")


def rename_to_jpg(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            # 分离文件名和后缀
            base_name, _ = os.path.splitext(file)
            
            # 构造新的文件名和路径
            new_file = base_name + ".jpg"
            old_path = os.path.join(root, file)
            new_path = os.path.join(root, new_file)
            
            # 重命名文件
            os.rename(old_path, new_path)
            print(f"重命名: {file} -> {new_file}")


def visualization():
    IMAGE_INPUT_PATH = '/workspace/YOLOv8-CL-main/VOCdevkit/VOC2007/JPEGImages'
    XML_INPUT_PATH = '/workspace/YOLOv8-CL-main/VOCdevkit/VOC2007/Annotations'
    IMAGE_OUTPUT_PATH = '/workspace/YOLOv8-CL-main/visual-images-1114'

    imglist = os.listdir(IMAGE_INPUT_PATH)
    xmllist = os.listdir(XML_INPUT_PATH)

    for i in range(len(imglist)):
        # 每个图像全路径
        image_input_fullname = IMAGE_INPUT_PATH + '/' + imglist[i]
        xml_input_fullname = XML_INPUT_PATH + '/' + str(imglist[i]).split('.')[0] + '.xml'
        image_output_fullname = IMAGE_OUTPUT_PATH + '/' + imglist[i]

        # print(f"image is {image_input_fullname}, xml is {xml_input_fullname}")

        image = cv2.imread(image_input_fullname)

        # 解析XML文件
        tree = ET.parse(xml_input_fullname)
        root = tree.getroot()

        # 获取图像尺寸
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)

        # 遍历所有目标
        for obj in root.findall('object'):
            # 获取目标类别和边界框坐标
            label = obj.find('name').text
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)

            # 选择颜色
            color = (0, 255, 0)  # 绿色  

            # 在图像上画出边界框和类别标签
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 4)
            cv2.putText(image, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 12)

            """
            # 直接查看生成结果图 
            cv2.imshow('show', img)
            cv2.waitKey(0)
            """

        cv2.imwrite(image_output_fullname, image)



def visual_one(imgfile, xmlfile):
    IMAGE_OUTPUT_PATH = '/workspace/YOLOv8-CL-main/visual-images-10'
    image_output_fullname = IMAGE_OUTPUT_PATH + '/' + imgfile.split('/')[-1]

    image = cv2.imread(imgfile)
    img_name = imgfile.split('/')[-1]
    shutil.copy2(imgfile, os.path.join(IMAGE_OUTPUT_PATH, img_name.split('.')[0] + '+_out.jpg')

    # 解析XML文件
    tree = ET.parse(xmlfile)
    root = tree.getroot()

    # 获取图像尺寸
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    # 遍历所有目标
    for obj in root.findall('object'):
        # 获取目标类别和边界框坐标
        label = obj.find('name').text
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)

        # 选择颜色
        color = (0, 255, 0)  # 绿色  

        # 在图像上画出边界框和类别标签
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 4)
        cv2.putText(image, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 4)

        """
        # 直接查看生成结果图 
        cv2.imshow('show', img)
        cv2.waitKey(0)
        """

    cv2.imwrite(image_output_fullname, image)
    shutil.copy2(xmlfile, os.path.join(IMAGE_OUTPUT_PATH, xmlfile.split('/')[-1]))



def extract_and_visual_pics(root_folder):
    # target_names = ['yw_gkxfw', 'bjzc', 'ddjt', 'jsxs_ddjt', 'yx', 
    #                 'pzq', 'cysb_tg', 'jsxs_ddyx', 'yxdgsg', 
    #                 'jdyxx', 'drq', 'cysb_lqq', 'cysb_cyg', 'xmbhyc',
    #                 'wcgz', 'wcaqm', 'jyz_pl', 'yw_nc', 'xmbhzc',
    #                 'bj_bpmh', 'bjdsyc_zz', 'bj_bpps', 'jsxs_jdyxx', 'bmwh',
    #                 'jyh', 'yljdq', 'sly_bjbmyw', 'gcc_mh', 'sly_dmyw', 
    #                 'ecjxh', 'hxq_gjbs', 'ws_ywzc', 'ywzt_yfyc', 'ywzt_yfzc'
    #                 'bj_wkps', 'hxq_gjzc', 'qtjdq', 'ws_ywyc', 'aqmzc',
    #                 'kgg_ybf', 'kgg_ybh', 'fhz_h', 'gzzc', 'zsd_l',
    #                 'zsd_m', 'bjdsyc_sx', 'gcc_ps', 'cysb_qyb', 'hxq_gjtps',
    #                 'fhz_f', 'kk_h', 'kk_f', 'yljdq_flow', 'yljdq_stop', 
    #                 'jyhbx', 'hxq_yfps', 'hzyw', 'mbhp', 'fhz_ztyc']
    target_names = ['sly_bjbmyw', 'cysb_cyg', 'cysb_tg', 'jyh', 'qtjdq',
                    'bjzc', 'jdyxx', 'ddjt', 'sly_dmyw', 'cysb_lqq']
    img_root = '/workspace/YOLOv8-CL-main/VOCdevkit/VOC2007/JPEGImages'
    name_counter = Counter()

    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".xml"):
                file_path = os.path.join(root, file)
                
                # 解析XML文件
                tree = ET.parse(file_path)
                root_element = tree.getroot()
                
                # 查找所有<object><name>标签
                for obj in root_element.findall(".//object"):
                    name = obj.find("name")
                    if name is not None and name.text in target_names:
                        if(name_counter[name.text] > 10):
                            continue
                        
                        name_counter[name.text] += 1
                        # find xml and images
                        img = file.split('.')[0] + '.jpg'
                        img_path = os.path.join(img_root, img)
                        # visual them and save them
                        visual_one(img_path, file_path)


if __name__ == '__main__':
    # count classes
    # root_folder = "/workspace/YOLOv8-CL-main/VOCdevkit/VOC2007/Annotations"  # 替换为你的根文件夹路径
    # output_file = "number_classes_sg_2.txt"
    # count_object_names_in_xml(root_folder, output_file)
    # print(f"统计结果已保存到 {output_file}")

    # Generate datasets
    # root_folder = "/workspace/data_thu/dataset"        # 替换为你的根文件夹路径
    # annotation_folder = "/workspace/YOLOv8-CL-main/VOCdevkit/VOC2007/Annotations"    # 替换为Annotation文件夹路径
    # images_folder = "/workspace/YOLOv8-CL-main/VOCdevkit/VOC2007/JPEGImages"            # 替换为Images文件夹路径

    # organize_files(root_folder, annotation_folder, images_folder)


    # split dataset
    # dataset_foler = "/workspace/YOLOv8-CL-main/VOCdevkit/VOC2007"
    # split_dataset(dataset_foler)

    # clean filenames
    # folder_path = "/workspace/YOLOv8-CL-main/VOCdevkit/VOC2007/Annotations"  # 替换为你的文件夹路径
    # rename_files_in_folder(folder_path)

    # clean xml files
    folder_path = "/workspace/YOLOv8-CL-main/VOCdevkit/VOC2007/Annotations"
    clean_xml(folder_path)

    # visual
    # visualization()

    # rename images
    # folder_path = "/workspace/YOLOv8-CL-main/VOCdevkit/VOC2007/JPEGImages"
    # rename_to_jpg(folder_path)

    # extract and visual pics
    # folder_path = "/workspace/YOLOv8-CL-main/VOCdevkit/VOC2007/Annotations"
    # extract_and_visual_pics(folder_path)
    
