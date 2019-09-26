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
X_trainingSet,y_trainingSet  = np.load('BreastData/X_trainingSet.npy'), np.load('BreastData/y_trainingSet.npy')
X_testSet, y_testSet = np.load('BreastData/X_testSet.npy'), np.load('BreastData/y_testSet.npy')
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
for index in range(50):
    index = index + 1
    clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping = False,
           epsilon=1e-08, hidden_layer_sizes=(10, 10), learning_rate='constant',
           learning_rate_init=0.001, max_iter=index, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=False,
           solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=1,
           warm_start=False)
    clf.fit(X_trainingSet,y_trainingSet)
    hypothesis = clf.predict(X_testSet)
    errorValid.append(1 - np.mean(np.abs(hypothesis - y_testSet)))
    hypothesis = clf.predict(X_trainingSet)
    errorTrain.append(1 - np.mean(np.abs(hypothesis - y_trainingSet)))



plt.figure(2)
plt.plot(range(50) ,errorValid, label = "testing")
plt.plot(range(50) ,errorTrain, label = "training")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Training Performance vs. time')
plt.ylabel('Accuracy')
plt.xlabel('iterations')

    
    
    
#input in the best time point.
print(confusion_matrix(y_testSet, hypothesis))  
print(classification_report(y_testSet, hypothesis))  


model = clf
learningCurve = skplt.estimators.plot_learning_curve(model, X_trainingSet, y_trainingSet, title='ANN Learning Curve', cv=None, shuffle=False, random_state=None, train_sizes=None, n_jobs=1, scoring=None, ax=None, figsize=None, title_fontsize='large', text_fontsize='medium')
