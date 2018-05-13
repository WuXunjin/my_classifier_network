# -*- coding: utf-8 -*-
# @Time    : 18-5-7 下午12:25
# @Author  : Kim Luo
# @Email   : kim_luo_balabala@163.com
# @File    : VGG16_224.py
# @Software: PyCharm
from base_networks.vgg16_re import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

class VGG_16_224(nn.Module):

    def __init__(self):
        super(VGG_16_224, self).__init__()
        self.n_classes = 20
        self.embedding = vgg16_re()
        self.embedding.init_vgg16_params()  # 载入预训练参数
        # self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.fc = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                nn.PReLU(),
                                nn.Linear(4096, 4096),
                                nn.PReLU(),
                                nn.Linear(4096, self.n_classes)
                                )
        # self.output_score = nn.Tanh()
        # self.transformer = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])  # 224 resize

    def forward(self, x):
        feature = self.embedding(x)
        # feature = self.pool5(feature)
        # print(feature.size())
        feature = feature.view(feature.size()[0], -1)  # 变成(batchsize,512*7*7)的大小
        score = self.fc(feature)
        # score=self.output_score(score)
        return score

    def load_v1(self, checkpoint, load_fc=False):#暂时不知道这个写的是啥意思
        model_dict = self.state_dict()
        layers = [k for k, v in model_dict.items()]

        pretrained_dict = torch.load(checkpoint)
        keys = [k for k, v in pretrained_dict.items()]
        keys.sort()
        # keys = keys[2:-4] #load until conv5

        to_load = []
        for k in keys:
            if k not in model_dict:
                continue
            #            if 'conv5' in k or 'bn5' in k:
            #                continue
            if 'conv' in k:
                to_load.append(k)
            if 'fc' in k and load_fc:
                to_load.append(k)

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in to_load and k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def load_v2(self,checkpointFold, epoch):
        checkpoint_path='%s/checkpoint_%d.tar'%(checkpointFold,epoch)
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        train_stat = checkpoint['train_stat']
        val_stat = checkpoint['val_stat']
        self.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint train_stat '{}', val_stat '{}', (epoch {})"
              .format(np.mean(train_stat), np.mean(val_stat),checkpoint['epoch']))

    def save_v1(self, checkpointFold, epoch):
        filename = '%s/vgg16_re_%03i.pth.tar' % (checkpointFold, epoch)
        torch.save(self.state_dict(), filename)

    def save_v2(self,save_dir='.',epoch=0,train_stat=None,val_stat=None):#这个文件夹名字不用写最后一个'/'
        torch.save({
            'epoch': epoch,
            'arch': [],#待拓展
            'state_dict': self.state_dict(),
            'train_stat': train_stat,
            'val_stat':val_stat,
        }, '%s/checkpoint_%d.tar'%(save_dir,epoch) )

def get_VGG16_224_model():
    model = VGG_16_224()
    return model
