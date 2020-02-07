# -*- coding: utf-8 -*-
# __author: CQQ
# @file: Function.py
# @time: 2019 10 29
# @email: chenqq18@163.com
import numpy as np
from scipy.spatial.distance import pdist,squareform
import copy
from sklearn.decomposition import KernelPCA
from metric_learn import LMNN
from tkinter import _flatten

def LMNN_Metric(datamatrix,datalabel):
    Dis_Matrix = np.zeros((len(datalabel), len(datalabel)))
    lmnn = LMNN(k=5, learn_rate=1e-6)
    lmnn.fit(datamatrix,datalabel)
    metric_func = lmnn.get_metric()
    for i in range(len(datalabel)):
        for j in range(len(datalabel)):
            Dis_Matrix[i,j] = metric_func(datamatrix[i],datamatrix[j])
    return Dis_Matrix

def Load_Data(filename):
    Datasets = np.loadtxt(filename,delimiter=',')
    [InstanceNum,AttributeNum] = Datasets.shape
    DataMatrix = Datasets[:,:AttributeNum-1]
    DataLabel = (Datasets[:,AttributeNum-1]).astype(int)
    ClassNum = len(set(DataLabel))
    return DataMatrix,DataLabel,InstanceNum,AttributeNum,ClassNum

#Calculate distance between data
def Calcu_Dis(InputMatrix):
    """
    :param InputMatrix: attribute space
    :return: DistanceMatrix
    """
    X = pdist(InputMatrix,'euclidean')
    DistanceMatrix = squareform(X,force='no',checks=True)
    return DistanceMatrix

def Calcu_MahalaDis(InputMatrix):
    n_samples = len(InputMatrix)
    dis_Ma = np.zeros((n_samples,n_samples))
    md = pdist(InputMatrix, 'mahalanobis')
    dis_Ma[np.triu_indices(n_samples, 1)] = md
    dis_Ma[np.tril_indices(n_samples, -1)] = md
    return dis_Ma

def CompareLabel(DistanceMatrix,Train_Label):
    """
    Parameters
    ----------
    DistanceMatrix:Distance of train data
    Train_Label:train label

    Returns
    -------
    CompareMatrix:第一列存放样本i同类样本个数(包括自己)，第二列之后存放样本i的下标，后面存档同类近邻样本
    """
    Label_Index_Matrix = np.argsort(DistanceMatrix)
    NumOfInstance = len(Train_Label)
    CompareMatrix = np.zeros((NumOfInstance,NumOfInstance+1))
    for i in range(NumOfInstance):
        label_i = Train_Label[i]
        temp=0
        for j in range(0,NumOfInstance):
            if label_i == Train_Label[Label_Index_Matrix[i][j]]:
                CompareMatrix[i][j+1]=Label_Index_Matrix[i][j]
                temp = temp + 1
            else:
                CompareMatrix[i][0] = temp
                break
    CompareMatrix = CompareMatrix.astype(int)
    return  CompareMatrix


def Affinity_propagatio_Modify(CompareMatrix):
    Remain_Index = list(np.argsort(-CompareMatrix[:, 0]))
    Cluster_Checked = []
    while Remain_Index:
        Index_Checked,Cluster = [], []
        # 保证L中的索引是有序的
        CompareMatrix_T = copy.deepcopy(CompareMatrix[Remain_Index, :])
        Remain_Index = CompareMatrix_T[list(np.argsort(-CompareMatrix_T[:, 0])), 1]
        L = list(CompareMatrix[Remain_Index[0], 1:CompareMatrix[Remain_Index[0], 0] + 1])
        while L:
            Li_Num = CompareMatrix[L[0], 0]
            Index_Checked.append(L[0])
            Index_Cluster = list(set(CompareMatrix[L[0], 1:Li_Num + 1])-set(Index_Checked))
            L.extend(list(set(Index_Cluster)-set(L)))
            L.remove(L[0])
        fla_Cluster_Checked = list(_flatten(Cluster_Checked))
        #判断得到的类簇与已有的类簇之间是否存在交集
        if len(set(fla_Cluster_Checked)&set(Index_Checked)):
            for i in range(len(Cluster_Checked)):
                if len(set(Index_Checked)&set(Cluster_Checked[i])):
                    Cluster_Checked[i] = list(set(Cluster_Checked[i])|set(Index_Checked))
        else:
            Cluster_Checked.append(Index_Checked)
        Cluster.extend(Index_Checked)
        Remain_Index = list(set(Remain_Index)-set(Cluster))
    #重新处理Cluster_Checked
    L = list(range(len(Cluster_Checked)))
    for i in L:
        LL = list(filter(lambda x:x<i, L))
        for j in LL:
            if len(set(Cluster_Checked[j])&set(Cluster_Checked[i])):
                Cluster_Checked[j] = list(set(Cluster_Checked[i]) | set(Cluster_Checked[j]))
                L.remove(i)
                break
    Cluster_Checked = [Cluster_Checked[i] for i in L]
    return Cluster_Checked

def RecordIndexOfClass (DataLabel):
    NumOfClass = len(np.unique(DataLabel))
    Record_Matrix = []
    for i in range(1,NumOfClass+1):
        ClassIndex_i = np.where(DataLabel==i)
        ClassIndex_i = ClassIndex_i[0]
        Record_Matrix.append(ClassIndex_i)
    return Record_Matrix


def KPCA_APC(Data_Origi, DataLabel,n_class):
    kpca = KernelPCA(kernel='rbf', fit_inverse_transform=True, gamma=10,n_components=n_class*2)
    Data_trans = kpca.fit_transform(Data_Origi)
    Dis_Matrix = Calcu_MahalaDis(Data_trans)
    CompareMatrix = CompareNoiseLabel(Dis_Matrix, DataLabel)
    Cluster_Checked = Affinity_propagatio_Modify(CompareMatrix)
    return Cluster_Checked,Data_trans



def Count(Cluster_Checked,set_vlaue,n_samples):
    overlap = set([])
    for i in range(len(Cluster_Checked)):
        if len(Cluster_Checked[i]) < set_vlaue:
            overlap = overlap | set(Cluster_Checked[i])
    lap_ratio = len(list(overlap))/n_samples
    return lap_ratio