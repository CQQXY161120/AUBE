# -*- coding: utf-8 -*-
# __author: CQQ
# @file: Figure_to_Scatter.py
# @time: 2019 11 13
# @email: chenqq18@163.com
import numpy as np
from sklearn import preprocessing
from skimage import io, transform
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

# 读取原始图片
img0 = io.imread('X.png')
img_gray = rgb2gray(transform.rescale(img0, 0.5))  # 压缩比例
binarizer = preprocessing.Binarizer(threshold=.45).fit(img_gray)
img_gray_binary = binarizer.transform(img_gray)
scatter_idx = np.array(np.where(img_gray_binary == 0))
scatter_idx1 = np.array(np.where(img_gray_binary != 0))
dataset = np.zeros((3, scatter_idx.shape[1] + scatter_idx1.shape[1]))
dataset[:2, :scatter_idx.shape[1]] = np.array(np.where(img_gray_binary == 0))
dataset[2, :scatter_idx.shape[1]] = 1
dataset[:2, scatter_idx.shape[1]:] = np.array(np.where(img_gray_binary != 0))
dataset[2, scatter_idx.shape[1]:] = 2
dataset = dataset.transpose()
plt.figure()
plt.scatter(dataset[:,1], dataset[:,0], marker='.', c=dataset[:,2])
ax = plt.gca()
ax.invert_yaxis()
plt.axis('off')
plt.show()


