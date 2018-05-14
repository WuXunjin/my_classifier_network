# -*- coding: utf-8 -*-
# @Time    : 18-5-14 下午12:01
# @Author  : Kim Luo
# @Email   : kim_luo_balabala@163.com
# @File    : test_grad_cam.py
# @Software: PyCharm
'''这个函数主要是为了获取grad cam的结果，然后保存下来，当然循环的时候也会记录mIoU这些
所以也要实现测度的计算
初始化就载入模型，类别数21（包含背景），输出文件夹
从测试输入就是图像路径
会有一个接口给打印出当前的miou情况
'''
import lib.Experiment as ex
from datasets.VOC_dataset import VOC_dataset
import torch.nn as nn
import numpy as np
from sklearn.metrics import average_precision_score
from torch.autograd.variable import Variable
import torch.utils.data
import torchvision.transforms as transforms
class test_grad_cam():
    def __init__(self,ex_dir = './data/test_grad_cam_training_luo_v1',model=None,n_class=21):
        ex.check_dir(ex_dir)
        self.ex_dir=ex_dir
        self.model=model # 模型
        self.n_class=n_class