# -*- coding: utf-8 -*-
# __author: CQQ
# @file: Synthetic data generation.py
# @time: 2019 10 30
# @email: chenqq18@163.com
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)
n_samples = 1500
#%%
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,noise=.15)
X = noisy_circles
plt.figure('circles')
plt.scatter(X[0][:,0], X[0][:,1], s=10,c = X[1])
plt.show()

#%% Moon
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.15)
X = noisy_moons
plt.figure('Moon')
plt.scatter(X[0][:,0], X[0][:,1], s=10,c = X[1])
plt.show()
#%% Anisotropicly distributed data
random_state = 60
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.4, -0.5], [-0.4, 0.9]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)
plt.figure('aniso')
plt.scatter(X_aniso[:,0], X_aniso[:,1], s=10,c = y)
plt.show()

