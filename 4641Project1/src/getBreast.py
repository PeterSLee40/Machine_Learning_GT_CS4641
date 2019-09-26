# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 21:25:33 2018

@author: PeterLee
"""
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler
dtype={'user_id': float}
dataset = pd.read_csv("breastData.csv")  
X = dataset.iloc[:, 2:31].values  
y = dataset.iloc[:, 1].values
print [X, y]
X_trainingSet, X_testSet, y_trainingSet, y_testSet = train_test_split(X, y, test_size=.25)
scaler = StandardScaler()  
scaler.fit(X_trainingSet)
X_trainingSet = scaler.transform(X_trainingSet)  
X_testSet = scaler.transform(X_testSet)

