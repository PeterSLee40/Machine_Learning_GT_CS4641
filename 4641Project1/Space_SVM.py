# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 09:31:45 2018

@author: PeterLee
"""

from sklearn.svm import SVC
import scikitplot as skplt
import matplotlib.pyplot as plt
import numpy as np
from time import time
# construct the argument parse and parse the arguments
X_trainingSet,y_trainingSet  = np.load('SpaceData/X_trainingSet_unscale.npy'), np.load('SpaceData/y_trainingSet_unscale.npy')
X_testSet, y_testSet = np.load('SpaceData/X_testSet_unscale.npy'), np.load('SpaceData/y_testSet_unscale.npy')
from sklearn.metrics import classification_report, confusion_matrix  

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing

# train and evaluate a svm classifer on the raw pixel intensities

model = SVC(degree = 1)
skplt.estimators.plot_learning_curve(model, X_trainingSet, y_trainingSet, title='SVM Learning Curve', cv=None, shuffle=False, random_state=None, train_sizes=None, n_jobs=1, scoring=None, ax=None, figsize=None, title_fontsize='large', text_fontsize='medium')
model.fit(X_trainingSet, y_trainingSet)
start = time()
acctest = list()
acctrain = list()
times = list()
i = 10
for i in range(10, 100, 10):
    model = SVC(degree=1, max_iter = i)
    model.fit(X_trainingSet, y_trainingSet)
    pred_valid = model.predict(X_testSet)
    pred_train = model.predict(X_trainingSet)
    acctest.append(1 - np.mean(np.abs(pred_valid != y_testSet)))
    acctrain.append(1 - np.mean(np.abs(pred_train != y_trainingSet)))
    times.append(time()-start)

plt.figure(2)
train = plt.plot( range(10, 100, 100),acctrain, label='train')
test = plt.plot( range(10, 1000, 100),acctest, label='test')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.1)
plt.title('Training Performance over Iterations')
plt.ylabel('Accuracy')
plt.xlabel('Iterations')


errorValidSamples = []
learningCurve = skplt.estimators.plot_learning_curve(model, X_trainingSet, y_trainingSet, title='SVM Learning Curve', cv=None, shuffle=False, random_state=None, train_sizes=None, n_jobs=1, scoring=None, ax=None, figsize=None, title_fontsize='large', text_fontsize='medium')

model = SVC(degree = 1)
clf = model
clf = clf.fit(X_trainingSet, y_trainingSet)
hypothesis = clf.predict(X_testSet)
print(confusion_matrix(y_testSet, hypothesis))  
print(classification_report(y_testSet, hypothesis))