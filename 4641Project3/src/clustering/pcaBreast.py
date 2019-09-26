# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 21:51:12 2018

@author: PeterLee
"""
import sklearn.decomposition as decomp
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator

X_train,y_training  = np.load('../BreastData/X_trainingSet.npy'), np.load('../BreastData/y_trainingSet.npy')
X_test, y_test = np.load('../BreastData/X_testSet.npy'), np.load('../BreastData/y_testSet.npy')

#edit this to change number of components

pca = decomp.PCA(n_components=29)
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

plt.title('Cumulative variance ratio for Breast data')
plt.xlabel("Cumulative Variance")
plt.ylabel("Number of Components")