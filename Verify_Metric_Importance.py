# -*- coding: utf-8 -*-
# __author: CQQ
# @file: Verify_Metric_Importance.py
# @time: 2020 01 11
# @email: chenqq18@163.com
from metric_learn import LMNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

import Function as PF
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
datasets=['L1.txt','L2.txt','L3.txt']

set_vlaue=2
cv = 10

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(n_estimators=100),
    GaussianNB(),
    MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1),
]

Result_of_Upper = np.zeros([len(datasets),2])#LMNN和非度量学习
Result_of_acc_ave = np.zeros([len(datasets)*2,len(classifiers)])
Result_of_acc_std = np.zeros([len(datasets)*2,len(classifiers)])

for i in range(len(datasets)):
    print(datasets[i])
    new_path = os.path.join('.\data', datasets[i])
    Data_Origi, DataLabel, n_samples, n_attr, n_class = PF.Load_Data(new_path)
    #归一化处理
    scaler = MinMaxScaler()
    scaler.fit(Data_Origi)
    Data_Origi = scaler.transform(Data_Origi)
    for l in range(2):
        if l==0:
            #度量学习
            lmnn = LMNN(k=5, learn_rate=1e-6)
            lmnn.fit(Data_Origi, DataLabel)
            Data_trans = lmnn.transform(Data_Origi)
        else:
            Data_trans = Data_Origi
        #同质化融合
        Dis_Matrix = PF.Calcu_Dis(Data_trans)
        CompareMatrix = PF.CompareNoiseLabel(Dis_Matrix, DataLabel)
        Cluster_Checked = PF.Affinity_propagatio_Modify(CompareMatrix)
        lap_ratio = PF.Count(Cluster_Checked, set_vlaue, n_samples)
        Result_of_Upper[i,l] = 1- lap_ratio

        for j in range(len(classifiers)):
            print(classifiers[j])
            clf = classifiers[j]
            scores = cross_val_score(clf, Data_trans, DataLabel, cv=cv)
            Result_of_acc_ave[2*i+l, j] = scores.mean()
            Result_of_acc_std[2*i+l, j] = scores.std()