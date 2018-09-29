# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 13:49:18 2018

@author: PeterLee
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

#from sklearn.tree import DecisionTreeRegressor
#from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import AdaBoostClassifier
import scikitplot as skplt
from sklearn.metrics import classification_report, confusion_matrix  

import pandas as pd

# Create the dataset
rng = np.random.RandomState(1)
#X = np.linspace(0, 6, 100)[:, np.newaxis]
#y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])

X_trainingSet,y_trainingSet  = np.load('SpaceData/X_trainingSet_unscale.npy'), np.load('SpaceData/y_trainingSet_unscale.npy')
X_testSet, y_testSet = np.load('SpaceData/X_testSet_unscale.npy'), np.load('SpaceData/y_testSet_unscale.npy')

# Fit regression model
#regr_1 = DecisionTreeRegressor(max_depth=4)
errorValidDepth = []
errorTrainDepth = []
for i in range(1,9):
    i = i + 1
    clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=i),
                          n_estimators=300, random_state=rng)
    clf.fit(X_trainingSet, y_trainingSet)
    pred_valid = clf.predict(X_testSet)
    errorValidDepth.append(1 - np.mean(np.abs(pred_valid != y_testSet)))
    pred_train = clf.predict(X_trainingSet)
    errorTrainDepth.append(1 - np.mean(np.abs(pred_train != y_trainingSet)))
plt.plot(range(1,9,1), errorValidDepth)
plt.title("Accuracy vs. Classifier Depth")
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")

errorValidDepth = []
errorTrainDepth = []
for i in range(1,10):
    clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=3),
                          n_estimators = 100*i, random_state=rng)
    clf.fit(X_trainingSet, y_trainingSet)
    pred_valid = clf.predict(X_testSet)
    errorValidDepth.append(1 - np.mean(np.abs(pred_valid != y_testSet)))
    pred_train = clf.predict(X_trainingSet)
    errorTrainDepth.append(1 - np.mean(np.abs(pred_train != y_trainingSet)))
plt.plot(range(1,10), errorValidDepth)
plt.title("Accuracy vs. number of estimators")
plt.xlabel("number of estimators")
plt.ylabel("Accuracy")

clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=3),
                          n_estimators=300, random_state=rng)
skplt.estimators.plot_learning_curve(clf, X_trainingSet, y_trainingSet, title='Adaptive Boost Learning Curve', cv=None, shuffle=False, random_state=None, train_sizes=None, n_jobs=1, scoring=None, ax=None, figsize=None, title_fontsize='large', text_fontsize='medium')
plt.show()
clf.fit(X_trainingSet, y_trainingSet)

#y_1p = regr_1.predict(X_trainingSet)
hypothesis = clf.predict(X_testSet)
print(confusion_matrix(y_testSet, hypothesis))  
print(classification_report(y_testSet, hypothesis))  
