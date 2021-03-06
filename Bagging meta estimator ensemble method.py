# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 22:12:43 2020

@author: osehon 
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
import sklearn.model_selection as model_selection
from sklearn import tree
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

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

model = BaggingRegressor (n_estimators=10,random_state=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("y_pred:",y_pred)

pred = y_pred
true = y_test
################ Performance Evaluation #######################################
error = abs((true - pred)/pred)
percentage_error = error * 100
score = abs (r2_score(true, pred))
mse = mean_squared_error(true,pred)
mae = mean_absolute_error(true, pred)
print("R2:{0:.3f}, MSE:{1:.2f}, MAE:{1:.2f}, RMSE:{2:.2f}"
   .format(score, mse,mae,np.sqrt(mse)))
################ visualization #####################################
l = list(range(8)) #index numbers for x axis
l
fig = plt.figure()
ax = fig.add_subplot(111)

lns1 = ax.plot(l, y_pred, color =  "g", label = "Predicted values")
lns2 = ax.plot(l, y_test,color = "r", label = "True values")
ax2 = ax.twinx()
lns3 = ax2.plot(l, percentage_error, label = '% error')

# added these three lines
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

ax.grid()
ax.set_xlabel("trials")
ax.set_ylabel(r"true and predicted values ($μA/cm^2$)")
ax2.set_ylabel(r"% error")
plt.title('Bagging Ensemble Method') 
plt.show()

### Saving result in csv file
d = {'y_test':y_test, 'y_pred':y_pred,'error':error, 'percentage error':percentage_error}
prediction = pd.DataFrame(d, columns=None).to_csv('Bagging meem prediction.csv')
