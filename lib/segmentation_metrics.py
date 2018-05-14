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

def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise

    return height, width

def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")

'''
Exceptions
'''
class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
