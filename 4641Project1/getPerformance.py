# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 15:50:43 2018

@author: PeterLee
"""
import os
Data = ["Breast","Iris"]
Models = ["ANN", "Knn", "SVM", "DT", "Boost"]

for data in Data:
    for Model in Models:
        lol = data + "_" + Model + ".py"
        print lol
        execfile(lol)