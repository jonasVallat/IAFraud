
import pandas # Data analysis
import numpy as np # Multi-dimensional arrays and matrices

from matplotlib import pyplot as plt # MATLAB-like plotting framework
import seaborn as sn # Statistical data visualization based on matplotlib

import sklearn # Machine-learning library
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # Split data into train and test subse
from sklearn.metrics import classification_report # Printt the main metrics
from sklearn.metrics import roc_auc_score, roc_curve

import keras
from keras.models import Sequential
from keras.layers import Dense

data = pandas.read_csv('creditcard.csv') # Extracting data from csv
data = data.drop(['Time'],axis=1) # Removes irrelevent data 

# Split x/y data
x = data.iloc[:, data.columns != 'Class'] 
y = data.iloc[:, data.columns == 'Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=0) # Split test/training data

classifier = Sequential() # Initialising neural network

# Adding layers
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu', input_dim = 29))
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Compiling the neural network

classifier.fit(x_train, y_train, batch_size = 32, epochs = 2) # Training set

# Predicting the results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# Predict positive results probabilities
y_prob = classifier.predict_proba(x_test)  

# Viewing performances
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
df_cm = pandas.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7)) # Prepare display of figure
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g') # Set the figure data with the confusion matrix data frame
print("Accuracy: %0.4f" % sklearn.metrics.accuracy_score(y_test, y_pred))

plt.show()


# ROC Curve
probs_no_skill = [0 for _ in range(len(y_test))]

auc_no_skill = roc_auc_score(y_test, probs_no_skill)
auc_y = roc_auc_score(y_test, y_prob)

# Calculate roc curves
fpr_no_skill, tpr_no_skill, _ = roc_curve(y_test, probs_no_skill)
y_fpr, y_tpr, _ = roc_curve(y_test, y_prob)
# Plot the roc curve for the model
plt.plot(fpr_no_skill, tpr_no_skill, linestyle='--', label='No Skill')
plt.plot(y_fpr, y_tpr, marker='.', label='Predicted')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show the legend
plt.legend()

# Print the AUC scores
print('No Skill: ROC AUC=%.3f' % (auc_no_skill))
print('Logistic: ROC AUC=%.3f' % (auc_y))

# Show the plot
plt.show()