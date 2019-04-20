# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 14:52:58 2018

@author: PeterLee
"""

from sklearn.cluster import MiniBatchKMeans 
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from kneed import KneeLocator


X_train,y_train  = np.load('../BreastData/X_trainingSet.npy'), np.load('../BreastData/y_trainingSet.npy')
X_test, y_test = np.load('../BreastData/X_testSet.npy'), np.load('../BreastData/y_testSet.npy')
#edit the number of clusters desired
n_clusters = 20
n_components = n_clusters
#this is to find the elbows.
kmeans = MiniBatchKMeans (n_clusters =5).fit(X_train)
X_train_knn = kmeans.fit_predict(X_train)
X_train_knn = np.reshape(X_train_knn,(np.size(X_train_knn[:]),1))
X_test_knn = kmeans.predict(X_test)
X_test_knn = np.reshape(X_test_knn,(np.size(X_test_knn[:]),1))

#gets the bic of the test set,



# Plot the winner

#https://scikit-learn.org/stable/auto_examples/mixture/plot_knn_selection.html
from sklearn.neural_network import MLPClassifier
import scikitplot as skplt
from sklearn.metrics import classification_report, confusion_matrix  




errorTrain = []
errorValid = []
errorTrainknn = []
errorValidknn = []
#edit the number of iterations

x = range(100)
for index in x:
    index = index + 1
    clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping = False,
           epsilon=1e-08, hidden_layer_sizes=(10, 10), learning_rate='constant',
           learning_rate_init=0.001, max_iter=index, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=False,
           solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=1,
           warm_start=False)
    clfnn = clf
    clfnn.fit(X_train,y_train)
    hypothesis = clfnn.predict(X_test)
    errorValid.append(1 - np.mean(np.abs(hypothesis - y_test)))
    hypothesis = clfnn.predict(X_train)
    errorTrain.append(1 - np.mean(np.abs(hypothesis - y_train)))
    
    clf_knn = clf
    Xcomp = np.concatenate((X_train, X_train_knn), axis = 1)
    Xtest = np.concatenate((X_test, X_test_knn), axis = 1)
    clf_knn.fit(Xcomp, y_train)
    hypothesis_knn = clf_knn.predict(Xtest)
    errorValidknn.append(1 - np.mean(np.abs(hypothesis_knn - y_test)))
    hypothesis_knn = clf_knn.predict(Xcomp)
    errorTrainknn.append(1 - np.mean(np.abs(hypothesis_knn - y_train)))
    
    
    
plt.figure(2)
plt.plot(x ,errorValid, label = "testing")
plt.plot(x ,errorTrain, label = "training")
plt.title('Training Performance vs. time')
plt.ylabel('Accuracy')
plt.xlabel('iterations')
plt.plot(x ,errorValidknn, label = "testing knn")
plt.plot(x ,errorTrainknn, label = "training knn")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    
    
#input in the best time point.
#print(confusion_matrix(y_test, hypothesis))  
#print(classification_report(y_test, hypothesis))  


#model = clf_knn
#learningCurve = skplt.estimators.plot_learning_curve(model, Xcomp, y_train, title='ANN Learning Curve', cv=None, shuffle=False, random_state=None, train_sizes=None, n_jobs=3, scoring=None, ax=None, figsize=None, title_fontsize='large', text_fontsize='medium')
#learningCurve = skplt.estimators.plot_learning_curve(clf, X_train, y_train, title='ANN Learning Curve', cv=None, shuffle=False, random_state=None, train_sizes=None, n_jobs=3, scoring=None, ax=None, figsize=None, title_fontsize='large', text_fontsize='medium')
