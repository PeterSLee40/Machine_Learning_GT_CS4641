# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 11:31:01 2018

@author: PeterLee
"""

from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt  
import scikitplot as skplt
import sys
from StringIO import StringIO
X_trainingSet,y_trainingSet  = np.load('SpaceData/X_trainingSet_unscale.npy'), np.load('SpaceData/y_trainingSet_unscale.npy')
X_testSet, y_testSet = np.load('SpaceData/X_testSet_unscale.npy'), np.load('SpaceData/y_testSet_unscale.npy')

from sklearn.metrics import classification_report, confusion_matrix  



clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping = False,
       epsilon=1e-08, hidden_layer_sizes=(10, 10), learning_rate='constant',
       learning_rate_init=0.001, max_iter=1, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=False,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=1,
       warm_start=True)


errorTrain = []
errorValid = []
for index in range(1,100,10):
    clf = MLPClassifier(activation='relu',hidden_layer_sizes=(100,100,100), max_iter = index)
    clf.fit(X_trainingSet,y_trainingSet)
    hypothesis = clf.predict(X_testSet)
    errorValid.append(1 - np.mean(np.abs(hypothesis != y_testSet)))
    hypothesis = clf.predict(X_trainingSet)
    errorTrain.append(1 - np.mean(np.abs(hypothesis != y_trainingSet)))
plt.figure(2)
plt.plot( range(1,100,10),errorValid, label = "training")
plt.plot( range(1,100,10),errorTrain, label = "testing")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.1)
plt.title('Training Performance vs. iterations')
plt.ylabel('Accuracy')
plt.xlabel('size')

errorTest = []
errorTrain = []

index = 100
range2 = range(10, 100, 10)
for i in range2:
    i = i + 10
    clf = MLPClassifier(activation='relu',hidden_layer_sizes=(500,i))
    clf.fit(X_trainingSet,y_trainingSet)
    hypothesis = clf.predict(X_testSet)
    errorTrain.append(1 - np.mean(np.abs(hypothesis != y_testSet)))
    hypothesis = clf.predict(X_trainingSet)
    errorTest.append(1 - np.mean(np.abs(hypothesis != y_trainingSet)))
    
plt.figure(3)
plt.plot(range2,errorTest, label = "training")
plt.plot(range2,errorTrain, label = "testing")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Training Performance vs. neural network size')
plt.ylabel('Accuracy')
plt.xlabel('size')
clf = MLPClassifier(activation='relu',hidden_layer_sizes=(300, 40))
clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=1000)
skplt.estimators.plot_learning_curve(clf, X_trainingSet, y_trainingSet, title='ANN Learning Curve', cv=None, shuffle=False, random_state=None, train_sizes=None, n_jobs=1, scoring=None, ax=None, figsize=None, title_fontsize='large', text_fontsize='medium')

clf = clf.fit(X_trainingSet, y_trainingSet)
hypothesis = clf.predict(X_testSet)
print(confusion_matrix(y_testSet, hypothesis))  
print(classification_report(y_testSet, hypothesis))  