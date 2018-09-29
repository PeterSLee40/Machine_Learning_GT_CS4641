# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 21:28:42 2018

@author: PeterLee
"""

import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler
dtype={'user_id': float}
dataset = pd.read_csv("SpaceData/Skyserver_SQL2_27_2018 6_51_39 PM.csv")  
X = dataset.iloc[:, range(17)].values  
y = dataset.iloc[:, 17].values
X_trainingSet, X_testSet, y_trainingSet, y_testSet = train_test_split(X, y, test_size=.25)
#scaler = StandardScaler()  
#scaler.fit(X_trainingSet)
#X_trainingSet = scaler.transform(X_trainingSet)  
#X_testSet = scaler.transform(X_testSet)

