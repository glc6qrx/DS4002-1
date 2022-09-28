"""
Train deep learning model to predict age.

Datase from here: https://susanqq.github.io/UTKFace/
"""

#################################################
# Import packages
#################################################

import tensorflow
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,BatchNormalization,Flatten,Input
from sklearn.model_selection import train_test_split


#################################################
# Loop through age file names to extract ages
#################################################
path = "/home/cas7kvf/DS4002-1/data/UTKFace"


images = []
age = []

for img in os.listdir(path):
  ages = img.split("_")[0]
  img = cv2.imread(str(path)+"/"+str(img))
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  img = img / 255
  images.append(np.array(img))
  age.append(np.array(ages))
  
age = np.array(age,dtype=np.int64)
images = np.array(images)   #Forgot to scale image for my training. Please divide by 255 to scale. 


x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(images, age, random_state=42)


##################################################
# Define age model and train. 
#################################################

age_model = Sequential()
age_model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(200,200,3)))
#age_model.add(Conv2D(128, kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, strides=2))

age_model.add(Conv2D(128, kernel_size=3, activation='relu'))
#age_model.add(Conv2D(128, kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, strides=2))
              
age_model.add(Conv2D(256, kernel_size=3, activation='relu'))
#age_model.add(Conv2D(256, kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, strides=2))

age_model.add(Conv2D(512, kernel_size=3, activation='relu'))
#age_model.add(Conv2D(512, kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, strides=2))

age_model.add(Flatten())
age_model.add(Dropout(0.2))
age_model.add(Dense(512, activation='relu'))

age_model.add(Dense(1, activation='linear', name='age'))
              
age_model.compile(optimizer='adam', loss='mse', metrics=['mae', "acc"])
print(age_model.summary())              
                           
history_age = age_model.fit(x_train_age, y_train_age,
                        validation_data=(x_test_age, y_test_age), epochs=50)

age_model.save('age_model_50epochs.h5')



############################################################
# Evaluate performance of trained model (Accuracy and Loss)
############################################################

history = history_age

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('TrainingLoss.png')
plt.clf()

acc = history.history['acc']
#acc = history.history['accuracy']
val_acc = history.history['val_acc']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig('TrainingAccuracy.png')
plt.clf()

####################################################################
# Evaluate performance of trained model (Confusion Matrix)
####################################################################
from keras.models import load_model
#Test the model
my_model = load_model('age_model_50epochs.h5', compile=False)


predictions = my_model.predict(x_test_age)
y_pred15 = (predictions>= 0.15).astype(int)[:,0]
y_pred30 = (predictions>= 0.30).astype(int)[:,0]
y_pred50 = (predictions>= 0.25).astype(int)[:,0]

from sklearn import metrics
print ("Accuracy train 15 = ", metrics.accuracy_score(y_test_age, y_pred15))
print ("Accuracy train 30 = ", metrics.accuracy_score(y_test_age, y_pred30))
print ("Accuracy train 50 = ", metrics.accuracy_score(y_test_age, y_pred50))


#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm15=confusion_matrix(y_test_age, y_pred15)  
cm15df = pd.DataFrame(cm15)
cm15df.to_csv('cm15.csv')
cm30=confusion_matrix(y_test_age, y_pred30)  
cm30df = pd.DataFrame(cm30)
cm30df.to_csv('cm30.csv')
cm50=confusion_matrix(y_test_age, y_pred50)  
cm50df = pd.DataFrame(cm50)
cm50df.to_csv('cm50.csv')


########################################################
#Test On NBA
path = "/home/cas7kvf/DS4002-1/data/NBA Testing Data"

images_nba = []
age_nba = []

for img in os.listdir(path):
  ages = img.split("_")[0]
  img = cv2.imread(str(path)+"/"+str(img))
  img = img.reshape(1,200,200,3)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = img / 255
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


