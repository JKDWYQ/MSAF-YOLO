# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple grad-cam == 1.4.8
import logging
import os
import shutil
import sys
import time
from datetime import datetime

import thop
import torch
from prettytable import PrettyTable
from thop import profile
from torch import nn

from ultralytics import YOLO


def Flops(model_name='runs/MSAAFYOLO/yolov8m-MSAAFYOLO.yaml', imgsz=640):
    batch_size, height, width = 8, imgsz, imgsz
    model = YOLO(model_name).model  # select your model.pt path
    model.fuse()
    input = torch.randn(batch_size, 4, height, width)  # 4通道
    total_flops, total_params, layers = profile(model, [input], verbose=True, ret_layer_info=True)
    FLOPs, Params = thop.clever_format([total_flops * 2 / batch_size, total_params], "%.3f")
    table = PrettyTable()
    table.title = f'Model Flops:{FLOPs} Params:{Params}'
    table.field_names = ['Layer ID', "FLOPs", "Params"]
    for layer_id in layers['model'][2]:
        data = layers['model'][2][layer_id]
        FLOPs, Params = thop.clever_format([data[0] * 2 / batch_size, data[1]], "%.3f")
        table.add_row([layer_id, FLOPs, Params])
    return table


class Logger(object):
    num = 0

    def __init__(self, filename='default.log', stream=sys.stdout, batch_num=875, add_flag=True):
        self.terminal = stream
        print("filename:", filename)
        self.filename = filename
        self.add_flag = add_flag
        self.batch_num = batch_num
        # self.log = open(filename, 'a+')

    def write(self, message):
        find_batch_num = message.find(f'/{self.batch_num}') != -1
        find_INFO = message.find(f'[INFO]') != -1
        self.add_flag = False if find_batch_num or find_INFO else True
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)  # 输出到控制台
                # log.write('2')  # 输出到log文件

    def flush(self):
        pass


def pretrain(model_name='yolov8m-MSAAFYOLO.yaml', project_path='runs/MSAAFYOLO', label_name=None, imgsz=640):
    index_yaml = model_name.find('.yaml')
    model_name = model_name if index_yaml == -1 else model_name[:index_yaml]
    if label_name is None:
        now = datetime.now()  # 得到时间字符串
        label_name = f'{now.year}{now.month}{now.day}{now.hour}'  # {now.minute} {now.second}
    if not os.path.exists(f'{project_path}/{model_name}_{label_name}'):
        os.makedirs(f'{project_path}/{model_name}_{label_name}')
    source_file = f'{project_path}/{model_name}.yaml'
    dest_path = f'{project_path}/{model_name}_{label_name}/{model_name}_{label_name}.yaml'
    if os.path.exists(f'{project_path}/{model_name}_{label_name}/{model_name}_{label_name}.yaml'):
        print(f'网络结构文件已存在, 确定覆盖{model_name}_{label_name}.yaml？(press any key to continue)')
        input()
    shutil.copy(source_file, dest_path)  # 保存此时网络结构文件
    source_file2 = f'ultralytics/nn/modules/DCMSA.py'
    dest_path2 = f'{project_path}/{model_name}_{label_name}/MSAAF_{label_name}.py'
    shutil.copy(source_file2, dest_path2)  # 保存此时网络代码文件
    ostdout = sys.stdout
    ostderr = sys.stderr
    sys.stdout = Logger(dest_path, sys.stdout, batch_num=875)
    sys.stderr = Logger(dest_path, sys.stderr, batch_num=875)  # 配置日志输出到文件
    print(Flops(f'{project_path}/{model_name}.yaml', imgsz))  # 输出Flops结果
    sys.stdout = ostdout
    sys.stderr = ostderr
    return True


def val_box(model_path='runs/wyq/yolov8m-RGBD4ch-split-1.33-2468-2/weights/best.pt'):
    model = YOLO(model_path)
    metrics = model.val()  # 不需要传参，这里定义的模型会自动在训练的数据集上作评估
    print(metrics.box.map50)  # map50
    print(metrics.box.map75)  # map75
    print(metrics.box.map)


def val_seg(model_path='runs/wyq/yolov8m-LLVIP-default/weights/best.pt'):
    model = YOLO(model_path)
    metrics = model.val()  # 不需要传参，这里定义的模型会自动在训练的数据集上作评估
    print(metrics.seg.map50)  # map50
    print(metrics.seg.map75)  # map75
    print(metrics.seg.map)


# 参数
project = 'runs'
MSAF_YOLO = 'runs/MSAF_YOLO'
imgsz = 640
batch = 8
epochs = 100
train = True
pretrain_value = False
result_value = True
yaml_value = False
dataset = 'Snackbox.yaml'
protect_train = True

if __name__ == '__main__':
    now = datetime.now()  # {now.year}{now.month}{now.day}{now.hour}
    # model = YOLO(f'{MSAAFYOLO}/yolov8-seg-default.yaml')  # 训练
    # model.train(project=MSAAFYOLO, name=f'yolov8', epochs=100, batch=batch, device=0,
    #             imgsz=imgsz, close_mosaic=10, amp=False, exist_ok=True, patience=30, deterministic=True, data=f'E:/DATASETS/LLVIP/yolo-LLVIP.yaml')
    yaml_path = 'yolov8m-seg-default.yaml'
    # yaml_path = 'yolov8m-seg-default.yaml'
    dataset_label, yaml_name = dataset[:dataset.find('.yaml')], yaml_path[:yaml_path.find('.yaml')]
    label = f'{dataset_label}_{now.year}{now.month}{now.day}'  # 当前实验的标签
    continue_train = False  # 是否为重新训练 False or True
    if continue_train:
        file_name = 'yolov8m-seg_label'
        yaml_name = file_name[:file_name.find('_')]
        label = file_name[file_name.find('_') + 1:]
    if not protect_train:
        # 直接训练
        if not continue_train:
            model = YOLO(f'{MSAF_YOLO}/{yaml_name}.yaml')
            model.train(project=MSAF_YOLO, name=f'{yaml_name}_{label}', epochs=epochs, batch=batch, device=0, imgsz=imgsz, close_mosaic=10, amp=False,
                        exist_ok=True, save_period=100, patience=30, deterministic=True, data=f'C:/DATASETS_temp/MSAAFYOLO/{dataset}')
        else:
            model = YOLO(f'{MSAF_YOLO}/{yaml_name}_{label}/weights/last.pt')
            model.train(resume=True)  # 重新训练
    else:  # 检测是否满足训练条件
        pretrain_value = pretrain(yaml_name, MSAF_YOLO, label_name=f'{label}', imgsz=imgsz)  # 计算Flops等并保存 todo 注释
        result_value = os.path.exists(f'{MSAF_YOLO}/{yaml_name}_{label}/weights/best.pt') or os.path.exists(f'{MSAF_YOLO}/{yaml_name}_{label}/weights/last.pt')
        yaml_value = os.path.exists(f'{MSAF_YOLO}/{yaml_name}_{label}/{yaml_name}_{label}.yaml')
        train = False if pretrain_value or result_value or not yaml_value else True  # 保证只有本次pretrain没有运行 且网络不存在训练结果文件 且网络结构相关文件已保存时train才为真
        continue_value = True if not pretrain_value and result_value and yaml_value else False  # 本次pretrain没有运行 且存在训练文件 且相关文件已保存时continue_value才为真
        if not continue_train:
            print(f'----------------------本次将用{dataset}数据集训练{MSAF_YOLO}/{yaml_name}_{label}----------------------')
            print('准备进行网络训练\t\t√')
            print('注意:网络已经存在训练结果文件, 确定覆盖训练网络？\t口') if result_value else print('网络不存在训练结果文件\t√')
        else:
            print(f'----------------------本次将用对{MSAF_YOLO}/{yaml_name}_{label}/weights进行断点续训----------------------')
            print('准备进行断点续训\t\t√')
            print('Error:网络不存在训练结果文件,无法继续训练。\tX') if not result_value else print('网络已存在训练结果文件\t√')
        print('Warning:本次已运行pretrain, 确定直接训练网络？\t口') if pretrain_value else print('本次pretrain没有运行\t√')
        print('Warning:网络结构文件未保存, 确定不提前保存文件？\t口') if not yaml_value else print('网络结构相关文件已保存\t√')
        if continue_train:
            if not continue_value:
                print('请注意处理好上述问题。(press any key to continue)\t口\n手动确认继续断点续训\t√')
                input()
            print('将在五秒后开始断点续训')
            time.sleep(5)
            model = YOLO(f'{MSAF_YOLO}/{yaml_name}_{label}/weights/last.pt')
            model.train(resume=True)  # 重新训练
        else:
            if not train:
                print('请注意处理好上述问题。(press any key to continue)\t口')
                input()
                print('手动确认继续训练网络\t√')
            print('将在五秒后开始训练网络')
            time.sleep(5)
            model = YOLO(f'{MSAF_YOLO}/{yaml_name}.yaml')  # 训练
            model.train(project=MSAF_YOLO, name=f'{yaml_name}_{label}', epochs=epochs, batch=batch, device=0, imgsz=imgsz, close_mosaic=10, amp=False,
                        exist_ok=True, save_period=100, patience=30, deterministic=True, data=f'C:/DATASETS_temp/MSAAFYOLO/{dataset}')

