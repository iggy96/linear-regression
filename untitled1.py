# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 05:42:10 2020

@author: oseho
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import sklearn.model_selection as model_selection
from sklearn import metrics



data = pd.read_csv('training data_including_test_data_corrosion_rate_confirmation.csv')
data.head()
data.tail()
data.shape
#Plot each input against target
sns.pairplot(data, x_vars=['T', 'DO', 'S','pH','ORP'], y_vars='CR', size=7, aspect=0.7, kind='reg')
# use the list to select a subset of the original DataFrame
feature_cols = ['T', 'DO', 'S','pH','ORP']
X = data[feature_cols]
# print the first 5 rows of X
X.head()
# check the type and shape of X
print(type(X))
print(X.shape)
# select a Series from the DataFrame
y = data['CR']
# print the first 5 values
y.head()
# check the type and shape of Y
print(type(y))
print(y.shape)
#####################

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.95,test_size=0.05, random_state=1)
print ("X_train: ", X_train)
print ("y_train: ", y_train)
print("X_test: ", X_test)
print ("y_test: ", y_test)
###########################
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#####################
# instantiate
linreg = LinearRegression()
# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)
################################

y_pred = linreg.predict(X_test)
print("y_pred:",y_pred)

#######################################Evaluation metrics
pred = y_pred
true = y_test
#MAE
print("mean absolute error:", metrics.mean_absolute_error(true, pred))
#MSE
print("mean squared error:", metrics.mean_squared_error(true, pred))
#RMSE
print("root mean square error", np.sqrt(metrics.mean_squared_error(true, pred)))