# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 15:11:20 2018

@author: PeterLee, Plee99, GTID: 903309003
"""

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix  

dataset = pd.read_csv("pulsar_stars.csv")  
X = dataset.iloc[:, 0:7].values  
y = dataset.iloc[:, 8].values

X_trainingSet, X_testSet, y_trainingSet, y_testSet = train_test_split(X, y, test_size=0.25)  

scaler = StandardScaler()  
scaler.fit(X_trainingSet)

#regularizes the data, the mean becomes 0
X_trainingSet = scaler.transform(X_trainingSet)  
X_testSet = scaler.transform(X_testSet)

#picks the number of neighbors and then trains based on the training data.
classifier = KNeighborsClassifier(n_neighbors=7)  
classifier.fit(X_trainingSet, y_trainingSet)
#predicts based on the test set
hypothesis = classifier.predict(X_testSet)  
#prints out different metrics
print(confusion_matrix(y_testSet, hypothesis))  
print(classification_report(y_testSet, hypothesis))  
error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_trainingSet, y_trainingSet)
    pred_i = knn.predict(X_testSet)
    error.append(np.mean(pred_i != y_testSet))

#outputs a figure showing the error ass a function of K value
plt.figure(figsize = (13, 7))  
plt.plot(range(1, 40), error, color='blue', linestyle='dashed', marker='o',  
         markerfacecolor='red', markersize=10)
plt.title('Error Rate in the K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')  