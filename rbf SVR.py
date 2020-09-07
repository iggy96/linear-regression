# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 04:10:23 2020

@author: oseho
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import sklearn.model_selection as model_selection
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.svm import SVR


data = pd.read_csv('training data_including_test_data_corrosion_rate_confirmation.csv')
data.head()
data.tail()
data.shape
#Plot each input against target
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


###########SVR kernel types##############
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)

############################### RBF KERNEL ######################################
svr_rbf.fit(X_train, y_train)
y_pred = svr_rbf.predict(X_test)
print (y_pred)
pred = y_pred
true = y_test
####### rbf performance evaluation #######
error = abs(true - pred)
score = abs (r2_score(true, pred))
mse = mean_squared_error(true,pred)
mae = mean_absolute_error(true, pred)
print("R2:{0:.3f}, MSE:{1:.2f}, MAE:{1:.2f}, RMSE:{2:.2f}"
   .format(score, mse,mae,np.sqrt(mse)))
######### rbf visualization #############
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
plt.title('RBF SVR visualization') 
# show a legend on the plot 
plt.legend() 
# function to show the plot 
plt.show()

###################### LINEAR KERNEL ###################################
svr_lin.fit(X_train, y_train)
y_pred = svr_lin.predict(X_test)
print (y_pred)

pred = y_pred
true = y_test
##### linear kernel performance evaluation ####
error = abs(true - pred)
score = abs (r2_score(true, pred))
mse = mean_squared_error(true,pred)
mae = mean_absolute_error(true, pred)
print("R2:{0:.3f}, MSE:{1:.2f}, MAE:{1:.2f}, RMSE:{2:.2f}"
   .format(score, mse,mae,np.sqrt(mse)))
################ linear kernel visualization ######
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
plt.title('linear Kernel SVR visualization') 
# show a legend on the plot 
plt.legend() 
# function to show the plot 
plt.show()

###################### POLY KERNEL ###################################
svr_poly.fit(X_train, y_train)
y_pred = svr_poly.predict(X_test)
print (y_pred)

pred = y_pred
true = y_test
##### poly kernel performance evaluation ####
error = abs(true - pred)
score = abs (r2_score(true, pred))
mse = mean_squared_error(true,pred)
mae = mean_absolute_error(true, pred)
print("R2:{0:.3f}, MSE:{1:.2f}, MAE:{1:.2f}, RMSE:{2:.2f}"
   .format(score, mse,mae,np.sqrt(mse)))
################ poly kernel visualization ######
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
plt.title('Polynomial Kernel SVR visualization') 
# show a legend on the plot 
plt.legend() 
# function to show the plot 
plt.show()