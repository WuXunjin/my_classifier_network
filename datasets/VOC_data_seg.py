# -*- coding: utf-8 -*-
# @Time    : 18-5-6 下午1:37
# @Author  : Kim Luo
# @Email   : kim_luo_balabala@163.com
# @File    : VOC_data_seg.py
# @Software: PyCharm
import os
import os.path as osp
import numpy as np
from PIL import Image
import collections
from torch.utils import data


class VOCDataSet(data.Dataset):
    def __init__(self, split="train", img_transform=None, label_transform=None):
        self.split = split
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.files = []

        data_dir = '/home/kimy/data_sets/VOCdevkit/2012/VOC2012'
        imgsets_dir = osp.join(data_dir, "ImageSets/Segmentation/%s.txt" % split)
        with open(imgsets_dir) as imgset_file:
            for name in imgset_file:
                name = name.strip()
                img_file = osp.join(data_dir, "JPEGImages/%s.jpg" % name)
                label_file = osp.join(data_dir, "SegmentationClass/%s.png" % name)
                self.files.append({
                    "img": img_file,
                    "label": label_file
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')

        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return img, label

if __name__ == '__main__':
        loader=data.DataLoader(VOCDataSet(split="train")
                               ,batch_size=4, shuffle=True,num_workers=4, pin_memory=True)
        for a,b in enumerate(loader):
            print(a)
            print(b)