# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 11:52:26 2020

@author: oseho
"""
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge
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
alphas = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1,0.5, 1]
for a in alphas:
 model = Ridge(alpha=a, normalize=True).fit(X_train,y_train) 
 score = model.score(X, y)
 pred_y = model.predict(X)
 mse = mean_squared_error(y, pred_y) 
 print("Alpha:{0:.6f}, R2:{1:.3f}, MSE:{2:.2f}, RMSE:{3:.2f}"
    .format(a, score, mse, np.sqrt(mse)))

ridge_mod=Ridge(alpha=0.001, normalize=True).fit(X_train,y_train)
y_pred = ridge_mod.predict(X_test)
print("y_pred:",y_pred)
pred = y_pred
true = y_test
################ Performance Evaluation #######################################
error = abs(true - pred)
score = model.score(X_test,true)
mse = mean_squared_error(true,pred)
mae = metrics.mean_absolute_error(true, pred)
print("R2:{0:.3f}, MSE:{1:.2f}, MAE:{1:.2f}, RMSE:{2:.2f}"
   .format(score, mse,mae,np.sqrt(mse)))
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
