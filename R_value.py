# -*- coding: utf-8 -*-
# __author: CQQ
# @file: R_value.py
# @time: 2020 01 12
# @email: chenqq18@163.com

import Function as PF
import numpy as np
import os

datasets=['L1.txt','L2.txt','L3.txt']
RV = np.zeros([len(datasets),1])
for d in range(len(datasets)):
    print(datasets[d])
    new_path = os.path.join('.\data', datasets[d])
    Data_Origi, DataLabel, n_samples, n_attr, n_class = PF.Load_Data(new_path)
    Dis_Matrix = PF.Calcu_Dis(Data_Origi)
    Label_Index_Matrix = np.argsort(Dis_Matrix)
    Record_Matrix = PF.RecordIndexOfClass (DataLabel)
    count = 0
    for i in range(len(Record_Matrix)):
            Set_in = Record_Matrix[i]
            Set_out= set(range(n_samples))-set(Set_in)
            for j in Set_in:
                if len(set(Label_Index_Matrix[j,:8])&Set_out) -3 > 0:
                    count = count + 1
    R_value = 1 - count*1.0/n_samples
    RV[d] = R_value


