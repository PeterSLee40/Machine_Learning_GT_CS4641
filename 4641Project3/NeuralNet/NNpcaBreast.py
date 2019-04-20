# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 21:51:12 2018

@author: PeterLee
"""
import sklearn.decomposition as decomp
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator

X_train,y_train  = np.load('../BreastData/X_trainingSet.npy'), np.load('../BreastData/y_trainingSet.npy')
X_test, y_test = np.load('../BreastData/X_testSet.npy'), np.load('../BreastData/y_testSet.npy')
#edit the pca number
ncomp = 3;
'''
plt.plot(x,cumVar)
plt.axvline(knee, c = "r")
print("optimal knee is a k size of:"  , knee)
plt.title('Cumulative variance ratio for Breast data')
plt.xlabel("Cumulative Variance")
plt.ylabel("Number of Components")
'''
pca = decomp.PCA(n_components=ncomp, svd_solver = 'full')
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)

from sklearn.preprocessing import StandardScaler
Xscaler = StandardScaler()  
X_train_pca = Xscaler.fit_transform(X_train_pca)  
X_test_pca = Xscaler.transform(X_test_pca)

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 11:31:01 2018

@author: PeterLee
"""

from sklearn.neural_network import MLPClassifier
import scikitplot as skplt
from sklearn.metrics import classification_report, confusion_matrix  




errorTrain = []
errorValid = []
errorTrainpca = []
errorValidpca = []

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
    clf.fit(X_train,y_train)
    hypothesis = clf.predict(X_test)
    errorValid.append(1 - np.mean(np.abs(hypothesis - y_test)))
    hypothesis = clf.predict(X_train)
    errorTrain.append(1 - np.mean(np.abs(hypothesis - y_train)))
    
    clf_pca = clf
    Xcomp = X_train_pca
    Xtest = X_test_pca
    Xcomp = np.concatenate((X_train, X_train_pca), axis=1)
    Xtest = np.concatenate((X_test, X_test_pca), axis = 1)
    clf_pca.fit(Xcomp, y_train)
    hypothesis_pca = clf_pca.predict(Xtest)
    errorValidpca.append(1 - np.mean(np.abs(hypothesis_pca - y_test)))
    hypothesis_pca = clf_pca.predict(Xcomp)
    errorTrainpca.append(1 - np.mean(np.abs(hypothesis_pca - y_train)))
    
    
    
    
plt.figure(2)
plt.plot(x ,errorValid, label = "testing")
plt.plot(x ,errorTrain, label = "training")
plt.title('Training Performance vs. time')
plt.ylabel('Accuracy')
plt.xlabel('iterations')
plt.plot(x ,errorValidpca, label = "testing pca")
plt.plot(x ,errorTrainpca, label = "training pca")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    
    
#input in the best time point.
#print(confusion_matrix(y_testSet, hypothesis))  
#print(classification_report(y_testSet, hypothesis))  


#model = clf
#learningCurve = skplt.estimators.plot_learning_curve(model, X_trainingSet, y_trainingSet, title='ANN Learning Curve', cv=None, shuffle=False, random_state=None, train_sizes=None, n_jobs=1, scoring=None, ax=None, figsize=None, title_fontsize='large', text_fontsize='medium')
