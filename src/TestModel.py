#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 23:25:15 2022

@author: catherineschuster
"""
import tensorflow
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,BatchNormalization,Flatten,Input
from sklearn.model_selection import train_test_split
from keras.models import load_model

#Test the model
my_model = load_model('/home/cas7kvf/DS4002-1/age_model_50epochs.h5', compile=False)

#Test On NBA
path = "/home/cas7kvf/DS4002-1/data/NBA Testing Data"

images_nba = []
age_nba = []

for img in os.listdir(path):
  ages = img.split("_")[0]
  img = cv2.imread(str(path)+"/"+str(img))
  img = np.reshape(img, (-1, 200, 200, 3))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  images_nba.append(np.array(img))
  age_nba.append(np.array(ages))
  
y_test_nba = np.array(age_nba,dtype=np.int64)
x_test_nba = np.array(images_nba)   #Forgot to scale image for my training. Please divide by 255 to scale. 

#Test the model on NBA
nbapredictions = my_model.predict(x_test_nba)
y_pred_nba15 = (nbapredictions>= 0.15).astype(int)[:,0]
y_pred_nba30 = (nbapredictions>= 0.30).astype(int)[:,0]
y_pred_nba50 = (nbapredictions>= 0.50).astype(int)[:,0]

from sklearn import metrics
print ("Accuracy test 15 = ", metrics.accuracy_score(y_test_nba, y_pred_nba15))
print ("Accuracy test 30 = ", metrics.accuracy_score(y_test_nba, y_pred_nba30))
print ("Accuracy test 50 = ", metrics.accuracy_score(y_test_nba, y_pred_nba50))

#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm_nba15=confusion_matrix(y_test_nba, y_pred_nba15)  
cm_nba15df = pd.DataFrame(cm_nba15)
cm_nba15df.to_csv('cm_nba15.csv')
cm_nba30=confusion_matrix(y_test_nba, y_pred_nba30)  
cm_nba30df = pd.DataFrame(cm_nba30)
cm_nba30df.to_csv('cm_nba30.csv')
cm_nba50=confusion_matrix(y_test_nba, y_pred_nba50)  
cm_nba50df = pd.DataFrame(cm_nba50)
cm_nba50df.to_csv('cm_nba50.csv')
