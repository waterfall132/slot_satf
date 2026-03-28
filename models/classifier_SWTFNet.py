import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
import utils
import pdb
from abc import abstractmethod
from torch.nn.utils.weight_norm import WeightNorm



# =========================== Few-shot learning method: ProtoNet =========================== #
class Prototype_Metric(nn.Module):
    '''
        The classifier module of ProtoNet by using the mean prototype and Euclidean distance,
        which is also Non-parametric.
        "Prototypical networks for few-shot learning. NeurIPS 2017."
    '''
    def __init__(self, way_num=5, shot_num=5, neighbor_k=3):
        super(Prototype_Metric, self).__init__()
        self.way_num = way_num
        self.avgpool = nn.AdaptiveAvgPool2d(1)


    # Calculate the Euclidean distance between the query and the mean prototype of the support class.
    def cal_EuclideanDis(self, input1, input2):
        '''
         input1 (query images): 75 * 64 * 5 * 5
         input2 (support set):  25 * 64 * 5 * 5
        '''

        # input1---query images
        # query = input1.view(input1.size(0), -1)                                    # 75 * 1600     (Conv64F)
        query = self.avgpool(input1).squeeze(3).squeeze(2)                           # 75 * 64
        query = query.unsqueeze(1)                                                   # 75 * 1 * 1600 (Conv64F)


        # input2--support set
        input2 = self.avgpool(input2).squeeze(3).squeeze(2)                          # 25 * 64
        # input2 = input2.view(input2.size(0), -1)                                   # 25 * 1600
        support_set = input2.contiguous().view(self.way_num, -1, input2.size(1))     # 5 * 5 * 1600
        support_set = torch.mean(support_set, 1)                                     # 5 * 1600


        # Euclidean distances between a query set and a support set
        proto_dis = -torch.pow(query-support_set, 2).sum(2)                          # 75 * 5


        return proto_dis


    def forward(self, x1, x2):

        proto_dis = self.cal_EuclideanDis(x1, x2)

        return proto_dis



# =========================== Few-shot learning method: DN4 =========================== #
class ImgtoClass_Metric(nn.Module):

    def __init__(self, way_num=5, shot_num=5, neighbor_k=3):
        super(ImgtoClass_Metric, self).__init__()
        self.neighbor_k = neighbor_k
        self.shot_num = shot_num
        self.way_num=way_num

        self.threshold = 0.01
    def cal_cosinesimilarity(self, input1, input2):
        # Reshape and permute the tensors
        input1 = input1.contiguous().view(input1.size(0), input1.size(1), -1).permute(0, 2, 1)
        # print("input1.shape",input1.shape)#[75, 441, 64]
        input2 = input2.contiguous().view(input2.size(0), input2.size(1), -1).permute(0, 2, 1)
        # print("input2.shape",input2.shape)#[5, 441, 64]
        '''对query中的一部分局部描述符进行了剔除操作'''
        # L2 Normalization
        query = input1 / torch.norm(input1, 2, 2, True)
        # print("query.shape",query.shape)#[75, 441, 64]
        support_set = input2 / torch.norm(input2, 2, 2, True)#[5, 441, 64]/[25, 441, 64]
        support_set = support_set.contiguous ().view ( -1, self.shot_num * support_set.size ( 1 ),support_set.size ( 2 ) )  # 5 * 2205 * 64/5 * 441 * 64

        # Calculate class prototypes (mean of support set features)
        class_prototypes = torch.mean(support_set.unsqueeze(1), dim=1)
        # print ( "class_prototypes.shape", class_prototypes.shape )#5 * 2205 * 64/5 * 441 * 64
        prototype_similarity = torch.cosine_similarity(support_set.unsqueeze(1),class_prototypes,dim=-1)#[5, 1, 2205, 64]/[5, 1, 441, 64]和5 * 2205 * 64/5 * 441 * 64
        weights = F.softmax(prototype_similarity, dim=-1)  # Softmax over class prototypes for weighting
        # print("weights",weights)#weights.shape torch.Size([5, 5, 2205])/[5, 5, 441]

        aggregated_weights = torch.mean ( weights, dim=1 )  # 或者取平均值
        # print("aggregated_weights.shape",aggregated_weights.shape)#[5, 441]/[5, 2205]

        mean_weight = aggregated_weights.mean()
        std_weight = aggregated_weights.std()
        adaptive_threshold = mean_weight - std_weight
        # print("adaptive_threshold",adaptive_threshold)

        mask = aggregated_weights > adaptive_threshold
        # print("mask.shape",mask.shape)#[75, 441]
        masked_count = torch.sum(~mask)  # Count False in the mask
        total_vectors = mask.numel()  # Total number of vectors

        # print("Total number of vectors:", total_vectors)
        # print("Number of masked vectors:", masked_count.item())
        # 扩展掩码以匹配通道数
        expanded_mask = mask.unsqueeze ( -1 ).expand_as ( support_set )
        # print("expanded_mask.shape",expanded_mask.shape)#[75, 441, 64]
        # print("原版support_set的第一条向量：",support_set[0])
        support_set = support_set * expanded_mask.float ()
        # print("masked_support_set.shape",support_set.shape)#5 * 2205 * 64/5 * 441 * 64
        # print("masked_support_set的第一条向量：",support_set[0])


        class_prototypes = torch.mean(support_set.unsqueeze(1), dim=1)#5 * 2205 * 64/5 * 441 * 64
        # if self.shot_num==5:#修改查询重塑逻辑以匹配支持集的维度。这可能涉及到插值或重复查询集的特征，以匹配 5-shot 场景中支持集的更大特征集大小。
        #     query = query.repeat ( 1, 5, 1 )
        #query.unsqueeze(1):[75, 1, 441, 64]/[75, 1, 2205, 64]
        prototype_similarity = torch.cosine_similarity(query.unsqueeze(1),class_prototypes,dim=-1)#[75, 5, 441]/[75, 5, 2205]
        # print("prototype_similarity",prototype_similarity)
        weights = F.softmax(prototype_similarity, dim=-1)  # Softmax over class prototypes for weighting[75, 1, 5, 441]/[75, 1, 5, 2205]
        # print("weights",weights)

        aggregated_weights = torch.mean ( weights, dim=1 )  # 或者取平均值
        # print("aggregated_weights.shape",aggregated_weights.shape)#[75, 441]/[75, 2205]

        mean_weight = aggregated_weights.mean()
        std_weight = aggregated_weights.std()
        adaptive_threshold = mean_weight - std_weight
        # print("adaptive_threshold",adaptive_threshold)

        mask = aggregated_weights > adaptive_threshold##[75,441]/[75,  2205]
        # print("mask.shape",mask.shape)#[75, 441]
        masked_count = torch.sum(~mask)  # Count False in the mask
        total_vectors = mask.numel()  # Total number of vectors

        # print("Total number of vectors:", total_vectors)
        # print("Number of masked vectors:", masked_count.item())
        # 扩展掩码以匹配通道数
        expanded_mask = mask.unsqueeze ( -1 ).expand_as ( query )
        # print("expanded_mask.shape",expanded_mask.shape)#[75, 441, 64]/[75, 2205, 64]
        # print("原版query的第一条向量：",query[0])
        query = query * expanded_mask.float ()#搞半天都没有对query筛选后的进行使用
        # print("masked_query.shape",masked_query.shape)#[75, 2205, 64]/[75, 441, 64]
        # print("masked_query的第一条向量：",masked_query[0])

        # Apply weights to the original cosine similarity calculations
        innerproduct_matrix = torch.matmul(query.unsqueeze(1), support_set.permute(0, 2, 1))
        # print ( "innerproduct_matrix.shape", innerproduct_matrix.shape )

        # Top-k selection
        topk_value, _ = torch.topk(innerproduct_matrix, self.neighbor_k, 3)
        img2class_sim = torch.sum(torch.sum(topk_value, 3), 2)

        return img2class_sim

    def forward(self, x1, x2):
        return self.cal_cosinesimilarity(x1, x2)
