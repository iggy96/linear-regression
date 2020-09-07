# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 03:34:16 2020

@author: oseho
"""

from sklearn.linear_model import Lasso, LassoCV
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
import pandas as pd
import seaborn as sns
import sklearn.model_selection as model_selection
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

data = pd.read_csv('training data_including_test_data_corrosion_rate_confirmation.csv')
data.head()
data.tail()
data.shape
#Plot each input against target
#sns.pairplot(data, x_vars=['T', 'DO', 'S','pH','ORP'], y_vars='CR', size=7, aspect=0.7, kind='reg')
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

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.83,test_size=0.17, random_state=1)
model=Lasso().fit(X_train, y_train)
print(model)
Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)

y_pred = model.predict(X_test)
print("y_pred:",y_pred)
pred = y_pred
true = y_test
################ Performance Evaluation #######################################
error = abs(true - pred)
score = model.score(X_test,true)
mse = mean_squared_error(true,pred)
mae = metrics.mean_absolute_error(true, pred)
print("Alpha:{0:.2f}, R2:{1:.2f}, MSE:{2:.2f}, RMSE:{3:.2f}"
    .format(model.alpha, score, mse, np.sqrt(mse)))
################ visualization #####################################
l = list(range(8)) #index numbers for x axis
l
plt.plot(l, y_pred, label = "Predicted values") 
plt.plot(l, y_test, label = "True values") 
plt.plot(l, error, label = "error") 
# naming the x axis 
plt.xlabel('trials') 
# naming the y axis 
plt.ylabel('true and predicted values') 
# giving a title to my graph 
plt.title('ridge regression visualization') 
# show a legend on the plot 
plt.legend() 
# function to show the plot 
plt.show() 
