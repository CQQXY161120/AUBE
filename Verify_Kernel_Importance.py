# -*- coding: utf-8 -*-
# __author: CQQ
# @file: Verify_Kernel_Importance.py
# @time: 2020 01 12
# @email: chenqq18@163.com
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel, laplacian_kernel
import Function as PF
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
datasets=['L1.txt','L2.txt','L3.txt']

cv = 10

classifiers = [
    SVC(kernel='linear',C=0.025),
    SVC(kernel='poly', C=0.025),
    SVC(kernel='sigmoid', C=0.025),
    SVC(kernel='rbf', C=0.025),
    SVC(kernel=laplacian_kernel),
    SVC(kernel=chi2_kernel),
]

Result_of_acc_ave = np.zeros([len(datasets),len(classifiers)])
Result_of_acc_std = np.zeros([len(datasets),len(classifiers)])

for i in range(len(datasets)):
    print(datasets[i])
    new_path = os.path.join('.\data', datasets[i])
    Data_Origi, DataLabel, n_samples, n_attr, n_class = PF.Load_Data(new_path)
    scaler = MinMaxScaler()
    scaler.fit(Data_Origi)
    Data_Origi = scaler.transform(Data_Origi)
    for j in range(len(classifiers)):
        clf = classifiers[j]
        scores = cross_val_score(clf, Data_Origi, DataLabel, cv=cv)
        Result_of_acc_ave[i, j] = scores.mean()
        Result_of_acc_std[i, j] = scores.std()
