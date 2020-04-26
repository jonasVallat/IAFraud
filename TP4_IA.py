import pandas
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import keras
from keras.models import Sequential
from keras.layers import Dense

data = pandas.read_csv('creditcard.csv')    #extracting data

data = data.drop(['Time'],axis=1)   #removes irrelevent data

x = data.iloc[:, data.columns != 'Class']   #split x/y data
y = data.iloc[:, data.columns == 'Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=0)  #split test/training data

classifier = Sequential()   #Initialising neural network

#adding layers
classifier.add(Dense(units =15 , kernel_initializer = 'uniform', activation = 'relu', input_dim = 29))
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])    #compiling the neural network

classifier.fit(x_train, y_train, batch_size = 32, epochs = 2)   #training set

# Predicting the results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

#Viewing performances
print(classification_report(y_test, y_pred))

#Confusion Matrix
cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
df_cm = pandas.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
print("Accuracy: %0.4f" % sklearn.metrics.accuracy_score(y_test, y_pred))

plt.show()