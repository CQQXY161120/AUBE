# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:30:28 2020

@author: CQQ
"""
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
#%% L1
M, N = 3.5, 1500
Data,Label = make_blobs(n_samples=N ,n_features=2,centers=[[-M,M],[M,M]],
                     random_state=45,shuffle=False)
plt.scatter(Data[:,0], Data[:,1], c=Label, s=10)
plt.show()

#%% L2
from sklearn import cluster, datasets
n_samples = 1500
Data,Label = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
two_means = cluster.MiniBatchKMeans(n_clusters=2)
two_means.fit(Data)
y_pred = two_means.predict(Data)
plt.scatter(Data[:,0],Data[:,1],c=y_pred,s=10)
plt.show()

#%% L3
from sklearn import cluster, datasets
n_samples = 1500
Data,Label = datasets.make_moons(n_samples=n_samples, noise=.05)
two_means = cluster.MiniBatchKMeans(n_clusters=2)
two_means.fit(Data)
y_pred = two_means.predict(Data)
plt.scatter(Data[:,0],Data[:,1],c=y_pred,s=10)
plt.show()









