# -*- coding: utf-8 -*-
# @Time    : 18-5-13 下午12:34
# @Author  : Kim Luo
# @Email   : kim_luo_balabala@163.com
# @File    : VOC_dataset_aug.py
# @Software: PyCharm
'''针对多标签分类的任务，使用10582的增强训练集进行训练'''
import torch.utils.data as data
import os
import sys
import torchvision.datasets as datasets
import os.path as osp
from PIL import Image
import lib.Experiment as Ex
import numpy as np
import torch


def extract_label_from_gt(label_path):
    label_mask = Image.open(label_path).convert("P")
    gt_arry = np.asarray(label_mask)
    label = np.unique(gt_arry)
    label = label[label != 0]  # voc的标签中存在0表示背景，255表示忽略区域，都不是正常标签
    label = label[label != 255]
    # label_a=-1*np.ones(20)
    #现在改为0-1
    label_a = np.zeros(20)
    label_a[label-1]=1
    return label_a.tolist()  #输出1表示该类别存在，-1表示不存在


class VOC_dataset_aug(data.Dataset):
    '''train还是val必须是字符串'''

    def __init__(self, train='train', transform=None, label_transform=None):
        self.train = train
        self.img_transform = transform
        self.label_transform = label_transform
        self.files = []
        self.classes = ('__background__','aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        # imdb_save_dir = './imdb_save'  # 保存这个文件，用绝对路径
        imdb_save_dir = os.path.dirname(__file__)+'/imdb_save'
        if not os.path.exists(imdb_save_dir):
            os.mkdir(imdb_save_dir)
        imdb_save_path = imdb_save_dir + '/imdb_voc_multilabel_cls_AUG_'+ train+'.json'
        if osp.exists(imdb_save_path):
            self.files = Ex.load_json(imdb_save_path)
        else:  #TODO 可以考虑把验证集也加进来训练，或者是对验证集生成伪标签加入训练
            if train == 'train':
                img_dir='/home/kimy/data_sets/sbd/img'
                img_sets_txt='/home/kimy/data_sets/sbd/train_withoutval.txt'
                cls_gt_dir='/home/kimy/data_sets/sbd/cls'
            elif train == 'val':
                img_dir='/home/kimy/data_sets/VOCdevkit/2012/VOC2012/JPEGImages'
                img_sets_txt='/home/kimy/data_sets/VOCdevkit/2012/VOC2012/ImageSets/Segmentation/val.txt'
                cls_gt_dir = '/home/kimy/data_sets/VOCdevkit/2012/VOC2012/SegmentationClass'
            else:
                print('wrong train opt {}  !!!--- just train or val'.format(train))
                img_dir=''
                img_sets_txt=''
                cls_gt_dir=''
            with open(img_sets_txt) as imgset_file:
                for name in imgset_file:
                    name = name.strip()
                    img_file = img_dir+'/'+name+'.jpg'
                    label_file = cls_gt_dir + '/' + name + '.png'
                    cls_label = extract_label_from_gt(label_file)
                    self.files.append({
                        "img": img_file,
                        "label": cls_label  # 多标签分类的标签列表 #TODO 后面考虑是不是要转化为1,-1类似的向量
                    })
            Ex.save_json(obj=self.files, json_path=imdb_save_path)

    def __getitem__(self, index):  # TODO 未完成数据增广
        datafiles = self.files[index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')

        label = np.asarray(datafiles["label"])  # 有可能需要转换成numpy的数组
        label= torch.from_numpy(label)  # ai~ DoubleTensor but we need FloatTensor
        label=label.float()
        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return img, label

    def __len__(self):
        return len(self.files)

    def test_one_img(self,img_path='',if_cls_seg_gt=False):  #TODO 完成可视化展示
        img=Image.open(img_path).convert('RGB')
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img

if __name__ == '__main__':
    data = VOC_dataset_aug(train='val')
    print(0)