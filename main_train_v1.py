# -*- coding: utf-8 -*-
# @Time    : 18-5-6 下午5:00
# @Author  : Kim Luo
# @Email   : kim_luo_balabala@163.com
# @File    : main_train_v1.py
# @Software: PyCharm
'''发现写trainer函数，不如直接上，那么就在这里搞一下，
第一是数据集，第二是模型，第三是loss
不对，第一是实验路径，记录readme这些东西'''

import lib.Experiment as ex
from datasets.VOC_dataset import VOC_dataset
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

## 参数区 # TODO 到时候考虑做成命令行解析的参数那样
ex_dir='./data/training_luo_v1'
batch_size=24
workers=4
epoch_num=20
##
ex.check_dir(ex_dir)

#数据集
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) #TODO 这个是cifar10 里面的均值方差，先不着急用均值方差，其他的配置好再说
voc_train=VOC_dataset(train='train', transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]), label_transform=None)
voc_val=VOC_dataset(train='val', transform=transforms.Compose([transforms.ToTensor()]), label_transform=None)
train_loader = torch.utils.data.DataLoader(voc_train,
    batch_size=batch_size, shuffle=True,
    num_workers=workers, pin_memory=True)
val_loader = torch.utils.data.DataLoader(voc_val,
    batch_size=batch_size, shuffle=False,
    num_workers=workers, pin_memory=True)

#模型

# define loss function (criterion) and pptimizer
criterion = nn.CrossEntropyLoss().cuda()







