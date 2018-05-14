# -*- coding: utf-8 -*-
# @Time    : 18-5-14 下午12:14
# @Author  : Kim Luo
# @Email   : kim_luo_balabala@163.com
# @File    : segmentation_metrics.py
# @Software: PyCharm
'''
引用自[ https://github.com/BardOfCodes/seg_metrics_pytorch ]
[https://github.com/martinkersner/py_img_seg_eval/tree/c0bf9787ebbe3e5e2c7833efe78b5b2d392afaf1]
从这个代码里面引用一些函数过来
'''
import numpy as np
def mean_iou(hist_matrix, class_names):
    classes = len(class_names)
    class_scores = np.zeros((classes))
    for i in range(classes):
        class_scores[i] = hist_matrix[i,i]/(max(1,np.sum(hist_matrix[i,:])+np.sum(hist_matrix[:,i])-hist_matrix[i,i]))
        print('class',class_names[i],'miou',class_scores[i])
    print('Mean IOU:',np.mean(class_scores))
    return class_scores

def mean_pixel_accuracy(hist_matrix, class_names):
    classes = len(class_names)
    class_scores = np.zeros((classes))
    for i in range(classes):
        class_scores[i] = hist_matrix[i,i]/(max(1,np.sum(hist_matrix[i,:])))
        print('class',class_names[i],'mean_pixel_accuracy',class_scores[i])
    return class_scores

def pixel_accuracy(hist_matrix):
    num = np.trace(hist_matrix)
    p_a =  num/max(1,np.sum(hist_matrix).astype('float'))
    print('Pixel accuracy:',p_a)
    return p_a
