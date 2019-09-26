# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 14:17:39 2018

@author: PeterLee
"""
#http://scikit-learn.org/stable/modules/tree.html, skeleton code from
#http://dataaspirant.com/2017/02/01/decision-tree-algorithm-python-with-scikit-learn/
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt
import graphviz as gv
import numpy as np
import scikitplot as skplt
from sklearn.metrics import classification_report, confusion_matrix  


X_trainingSet,y_trainingSet  = np.load('BreastData/X_trainingSet.npy'), np.load('BreastData/y_trainingSet.npy')
X_testSet, y_testSet = np.load('BreastData/X_testSet.npy'), np.load('BreastData/y_testSet.npy')

clf = tree.DecisionTreeClassifier(max_depth = 7)
clf = clf.fit(X_trainingSet, y_trainingSet)
yp = clf.predict(X_trainingSet)


dot_data = tree.export_graphviz(clf, out_file=None) 
graph = gv.Source(dot_data) 
graph.render("LOL") 
graph

#find the optimal tree based off the number of nodes.
errorValidDepth = []
errorTrainDepth = []
maxDepth = range(10)
for i in maxDepth:
    clf = tree.DecisionTreeClassifier(max_depth = i + 1)
    clf.fit(X_trainingSet,y_trainingSet)
    pred_valid = clf.predict(X_testSet)
    errorValidDepth.append(1 - np.mean(np.abs(pred_valid != y_testSet)))
    pred_train = clf.predict(X_trainingSet)
    errorTrainDepth.append(1 - np.mean(np.abs(pred_train != y_trainingSet)))
plt.plot(maxDepth,errorTrainDepth, label = "training")
plt.plot(maxDepth,errorValidDepth, label = "testing")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.1)
plt.title('Training Performance over time')
plt.ylabel('Accuracy')
plt.xlabel('Depth')

import scikitplot as skplt
model = clf
learningCurve = skplt.estimators.plot_learning_curve(model, X_trainingSet, y_trainingSet, title='DT Learning Curve', cv=None, shuffle=False, random_state=None, train_sizes=None, n_jobs=1, scoring=None, ax=None, figsize=None, title_fontsize='large', text_fontsize='medium')

clf = tree.DecisionTreeClassifier(max_depth = 6)
clf.fit(X_trainingSet,y_trainingSet)
hypothesis = clf.predict(X_testSet)
print(confusion_matrix(y_testSet, hypothesis))  
print(classification_report(y_testSet, hypothesis))  
