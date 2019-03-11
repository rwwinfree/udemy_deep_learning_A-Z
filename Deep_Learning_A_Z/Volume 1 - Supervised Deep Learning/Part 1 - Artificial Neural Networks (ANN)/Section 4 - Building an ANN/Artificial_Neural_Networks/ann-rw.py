#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 13:26:26 2018

@author: ryanwinfree
"""

# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

###### Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Start with encoding the countries
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
# Next encode the genders (index 2)
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#Now create dummy variables to separate the encoded countries
#this will keep the NN from thinking there is a relation between the countries
#i.e. Spain = 2 and France = 0 but 2 is not > 0 in this case...they are unrelated
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#Remove a dummy variable to avoid falling into the "dummy variable trap"
#See http://www.algosome.com/articles/dummy-variable-trap-regression.html as a ref
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling This is Necessary!!!!
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


###### Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
## This is a classifier ANN

classifier = Sequential()

# Adding the input layer and the first hidden layer
## Step 1: Randomly initialize the weights to a value close to 0 but not 0
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation='relu'))

# Add the second idden layer
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
# if more than 1 dependent variable use activation = 'softmax'
classifier.add(Dense(units = 1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training Set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# Part 3 - Making the predictions and evaluating the model



# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) #returns true or false based on criteria
#This is removing the probablity value and instead setting the y_prediction 
#results to say if > 50% probability then say they will leave. If less then
#return that they stay

""" Homework Assignment Answer:
Predict if the customer with the following information will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
    
"""
# Adding the customer information as a 2d array and adding it already encoded
##new_prediction = classifier.predict(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
# Now we have to scale our data
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5) #returns true or false based on criteria



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


"""
#### Homework Assignment: Ryan W. Attempt...all wrong :( 

# Importing the dataset
dataset = pd.read_csv('test-data.csv')
x_example = dataset.iloc[:, 3:13].values
y_example = dataset.iloc[:, 13].values


# Encoding categorical data
# Start with encoding the countries
labelencoder_x_example_1 = LabelEncoder()
x_example[:, 1] = labelencoder_x_example_1.fit_transform(x_example[:, 1])
# Next encode the genders (index 2)
labelencoder_x_example_2 = LabelEncoder()
x_example[:, 2] = labelencoder_x_example_2.fit_transform(x_example[:, 2])

#Now create dummy variables to separate the encoded countries
#this will keep the NN from thinking there is a relation between the countries
#i.e. Spain = 2 and France = 0 but 2 is not > 0 in this case...they are unrelated
onehotencoder = OneHotEncoder(categorical_features = [1])
x_example = onehotencoder.fit_transform(x_example).toarray()

#Remove a dummy variable to avoid falling into the "dummy variable trap"
#See http://www.algosome.com/articles/dummy-variable-trap-regression.html as a ref
x_example = x_example[:, 1:]

"""




### Part 4 - Evaluating, Improving and Tuning the ANN

# Evaluating the ANN
##Need to combine keras and scikit together for this part
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score



# Improving the ANN

# Tuning the ANN





