# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 14:52:58 2018

@author: PeterLee
"""

from sklearn.cluster import MiniBatchKMeans 
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import sklearn.decomposition as decomp
from kneed import KneeLocator

X_train,y_training  = np.load('../BreastData/X_trainingSet.npy'), np.load('../BreastData/y_trainingSet.npy')
X_test, y_test = np.load('../BreastData/X_testSet.npy'), np.load('../BreastData/y_testSet.npy')

#edit the number of components for PCA
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
#plt.plot(x,cumVar)

knee = KneeLocator(x,cumVar,curve='convex', direction='increasing').knee
#plt.axvline(knee, c = "r")
print("optimal knee is a k size of:"  , knee)

#plt.title('Cumulative variance ratio for Breast data')
#plt.xlabel("Cumulative Variance")
#plt.ylabel("Number of Components")

pca = decomp.PCA(n_components=knee)
X_train_PCA = pca.fit_transform(X_train)
X_test_PCA = pca.transform(X_test)
#this is to find the elbows.
a = []
x = list(range(1,11));
for i in x:
    #increment clusters?
    nclusters = i + 1
    kmeans = MiniBatchKMeans (nclusters, random_state=1).fit(X_train_PCA)
    y_pred = kmeans.predict(X_test_PCA)
    #show bar chart of num clusters
    #inertia is theSum of squared distances
    a.append(kmeans.inertia_)
knee = KneeLocator(x,a,curve='convex', direction='decreasing').knee
print("optimal knee is a k size of:"  , knee)
plt.plot(x, a)
plt.axvline(knee, c = "r")
plt.ylabel("K means Inertia")
plt.xlabel("Number of Clusters")
plt.legend()
plt.show()