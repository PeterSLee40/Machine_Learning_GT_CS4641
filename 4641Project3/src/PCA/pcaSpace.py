# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 21:55:12 2018

@author: PeterLee
"""
import sklearn.decomposition as decomp
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator

X_train,y_training  = np.load('SpaceData/X_train.npy'), np.load('SpaceData/y_train.npy')
X_test, y_test = np.load('SpaceData/X_test.npy'), np.load('SpaceData/y_test.npy')
#edit the number of components desired
pca = decomp.PCA(n_components=15)
pca.fit(X_train)   
print (pca.explained_variance_ratio_ )
#pick the top 80% that gets most the data.
#plot cumulative pca
pcaVarRatioArray = pca.explained_variance_ratio_
cumVar = list()
for i in range(len(pca.explained_variance_ratio_)):
    total = pcaVarRatioArray[i]
    if (len(cumVar) > 0):
        total += cumVar[-1]
    cumVar.append(total)
x = range(len(pca.explained_variance_ratio_))
plt.plot(x,cumVar)
knee = KneeLocator(x,cumVar,curve='convex', direction='increasing').knee
plt.axvline(knee, c = "r")
print("optimal knee is a k size of:"  , knee)
plt.title('Cumulative variance ratio for Space data')
plt.ylabel("Cumulative Variance")
plt.xlabel("Number of Components")

