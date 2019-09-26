# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 15:11:20 2018

@author: PeterLee, Plee99, GTID: 903309003
"""

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
import scikitplot as skplt

from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix  

X_trainingSet,y_trainingSet  = np.load('SpaceData/X_trainingSet_unscale.npy'), np.load('SpaceData/y_trainingSet_unscale.npy')
X_testSet, y_testSet = np.load('SpaceData/X_testSet_unscale.npy'), np.load('SpaceData/y_testSet_unscale.npy')

#prints out different metrics


errorTest = []
errorTrain = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_trainingSet, y_trainingSet)
    pred_test = knn.predict(X_testSet)
    errorTest.append(1-np.mean(pred_test != y_testSet))
    pred_train = knn.predict(X_trainingSet)
    errorTrain.append(1-np.mean(pred_train != y_trainingSet))
plt.figure(figsize = (13, 7))  
plt.plot(range(1, 40), errorTest, color='blue', linestyle='dashed', marker='o',  
         markerfacecolor='red', markersize=10, label = "Test")
plt.plot(range(1, 40), errorTrain, color='red', linestyle='dashed', marker='x',  
         markerfacecolor='blue', markersize=10, label = "Training")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.2)
plt.title('Accuracy Rate in the K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Accuracy')  
plt.legend("test","training")
knn = KNeighborsClassifier(n_neighbors = 15)
knn.fit(X_trainingSet, y_trainingSet)
hypothesis = knn.predict(X_testSet)

print(confusion_matrix(y_testSet, hypothesis))  
print(classification_report(y_testSet, hypothesis))  
con_mat = confusion_matrix(y_testSet, hypothesis)*100/143
df_cm = pd.DataFrame(con_mat)
#sn.heatmap(df_cm, annot=True)


#outputs a figure showing the error ass a function of K value

knn = KNeighborsClassifier(n_neighbors = 15)
model = knn
learningCurve = skplt.estimators.plot_learning_curve(model, X_trainingSet, y_trainingSet, title='KNN Learning Curve', cv=None, shuffle=False, random_state=None, train_sizes=None, n_jobs=2, scoring=None, ax=None, figsize=None, title_fontsize='large', text_fontsize='medium')
