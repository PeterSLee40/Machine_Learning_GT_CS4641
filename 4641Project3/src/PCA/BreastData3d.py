# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 21:51:12 2018

@author: PeterLee
"""
import sklearn.decomposition as decomp
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator

X_train,y_train  = np.load('../BreastData/X_trainingSet.npy'), np.load('../BreastData/y_trainingSet.npy')
X_test, y_test = np.load('../BreastData/X_testSet.npy'), np.load('../BreastData/y_testSet.npy')


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn import decomposition


fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X_train)

#edit to generate figure for training set vs test set

#X = pca.transform(X_train)
#y = y_train


X = pca.transform(X_test)
y = y_test
# Reorder the labels to have colors matching the cluster results
    
colors = ['navy', 'turquoise', 'darkorange']
a = []
for i in y: 
    a.append(colors[i])
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=a, cmap=plt.cm.nipy_spectral,
           edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
#edit
plt.savefig('BreastData3dtest.png')

plt.show()