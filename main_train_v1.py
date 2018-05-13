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
import numpy as np
from sklearn.metrics import average_precision_score
from torch.autograd.variable import Variable
import torch.utils.data
import torchvision.transforms as transforms
import networks.VGG16_224 as vgg_model
from lib.loss import MultiLabelClsV1
import os
import networks.test as ts
# print( os.path.curdir)
# print(os.path.abspath(os.path.curdir))
# ts.print_cur_path()
## 参数区 # TODO 到时候考虑做成命令行解析的参数那样
def compute_mAP(labels,outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    for i in range(y_true.shape[0]):
        AP.append(average_precision_score(y_true[i],y_pred[i]))
    return np.mean(AP)

def test_one_img(model,dataset): #为了看看效果
    img_path='/home/kimy/data_sets/sbd/img/2007_000170.jpg'
    img=dataset.test_one_img(img_path)
    sz=list(img.shape)
    sz.insert(0,1)
    img = img.view(sz)
    img=Variable(img)
    model.eval()
    model.cuda()
    img=img.cuda()
    score=model(img)
    classes=dataset.classes
    print score
    print classes
    print 'done!'
ex_dir='./data/training_luo_v1'
batch_size=4
workers=4
epoch_num=20
##
ex.check_dir(ex_dir)

#数据集
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) #TODO 这个是cifar10 里面的均值方差，先不着急用均值方差，其他的配置好再说
voc_train=VOC_dataset(train='train', transform=transforms.Compose([transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]), label_transform=None )
voc_val=VOC_dataset(train='val', transform=transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()]),
                    label_transform=None  )
train_loader = torch.utils.data.DataLoader(voc_train,
    batch_size=batch_size, shuffle=True,
    num_workers=workers, pin_memory=True)
val_loader = torch.utils.data.DataLoader(voc_val,
    batch_size=batch_size, shuffle=False,
    num_workers=workers, pin_memory=True)

#模型 先用VGG的模型试试
model=vgg_model.get_VGG16_224_model()
#为了测试一下训练好的数据
# model.load_v2(checkpointFold=ex_dir,epoch=9)
# test_one_img(model,voc_train)

model.cuda()


# define loss function (criterion) and pptimizer
# criterion = MultiLabelClsV1(20)
criterion = nn.MultiLabelSoftMarginLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for t in range(10):
    model.train()
    mAP_train = []
    for i, (img, label) in enumerate(train_loader):
        img = img.cuda()
        label = label.cuda()
        label = Variable(label)
        input_img = Variable(img)

        output = model(input_img)
        model.zero_grad()
        mAP_train.append(compute_mAP(label.data, output.data))#计算准确率
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        print "train epoch %d step %d, loss=%.6f  mAp=%.4f" %(t, i, loss.data.cpu()[0],100 * np.mean(mAP_train))
    #测试
    model.eval()
    mAP_val = []
    for i, (img, label) in enumerate(val_loader):
        img = img.cuda()
        label = label.cuda()
        label = Variable(label)
        input_img = Variable(img)
        output = model(input_img)
        mAP_val.append(compute_mAP(label.data, output.data))  # 计算准确率
        loss = criterion(output, label)
        print "val epoch %d step %d, loss=%.4f  mAp=%.4f" %(t, i, loss.data.cpu()[0],100 * np.mean(mAP_val))
    print('epoch %d done!'%t)
    model.save_v2(save_dir=ex_dir,epoch=t,train_stat=mAP_train,val_stat=mAP_val)




