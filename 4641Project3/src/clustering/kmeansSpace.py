# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 13:19:57 2018

@author: PeterLee
"""
from sklearn.cluster import MiniBatchKMeans 
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from kneed import KneeLocator

X_train,y_training  = np.load('../SpaceData/X_trainingSet.npy'), np.load('../SpaceData/y_trainingSet.npy')
X_test, y_test = np.load('../SpaceData/X_testSet.npy'), np.load('../SpaceData/y_testSet.npy')


#this is to find the elbows.
a = []
#edit x to change the range of nclusters.
x = list(range(1,11));
for i in x:
    #increment clusters?
    nclusters = i + 1
    kmeans = MiniBatchKMeans (nclusters, random_state=1).fit(X_train)
    y_pred = kmeans.predict(X_test)
    #show bar chart of num clusters
    #inertia is theSum of squared distances
    a.append(kmeans.inertia_)
knee = KneeLocator(x,a,curve='convex', direction='decreasing').knee
plt.axvline(knee, c = "r")
print("optimal knee is a k size of:"  , knee)
plt.plot(x, a)
plt.legend()
plt.show()