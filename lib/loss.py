# -*- coding: utf-8 -*-
# @Time    : 18-5-7 下午12:40
# @Author  : Kim Luo
# @Email   : kim_luo_balabala@163.com
# @File    : loss.py
# @Software: PyCharm
# coding=utf-8
'''这个代码用来记录所有可能用到的loss'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelClsV1(nn.Module):
    """
    多标签分类损失，输入score是一维的向量，输入标签是1和-1的，1表示存在类别，-1表示不存在类别
    $$ t=\log^{1+\exp{(-c.*x)} } $$
    """

    def __init__(self, nClass):
        super(MultiLabelClsV1, self).__init__()
        self.nClass = nClass

    def forward(self,  output, label, size_average=True):
        #test version
        # a=-label*output#需要是doubleTensor 但是我的是FloatTensor
        # b=1.0+torch.exp( a )
        # losses=torch.log( b   )

        # losses = torch.log(1.0 + torch.exp(-label * output))  # 这样可能会出现inf
        '''
        上面这个写法可能会出现Inf
        参考matconvnet的写法：
        %t = log(1 + exp(-c.*X)) ;
        a = -c.*x ;
        b = max(0, a) ;
        t = b + log(exp(-b) + exp(a-b)) ;
        '''
        a=-label*output
        b = F.relu(a)
        losses = b + torch.log(torch.exp(-b) + torch.exp(a - b))
        return losses.mean() if size_average else losses.sum()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        # losses = 0.5 * (target.float() * distances +
        #                 (1 + -1 * target).float() * F.relu(self.margin - distances.sqrt()).pow(2))
        #这里有一个根号 distance使得损失的求导可能成为nan(我认为,所以把根号去掉)
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - distances).pow(2))

        return losses.mean() if size_average else losses.sum()

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)

if __name__ == '__main__':
    '''test'''
    loss=MultiLabelClsV1(3)
    a=torch.FloatTensor([10,-10,10])
    print(a)
    b=torch.FloatTensor([1,-1,1])
    print(b)
    print(loss(a,b))