# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 12:51:20 2018

@author: PeterLee
"""
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target
X_trainingSet, X_testSet, y_trainingSet, y_testSet = train_test_split(X, y, test_size=.25)
scaler = StandardScaler()  
scaler.fit(X_trainingSet)
X_trainingSet = scaler.transform(X_trainingSet)  
X_testSet = scaler.transform(X_testSet)


