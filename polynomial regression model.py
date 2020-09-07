from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import sklearn.model_selection as model_selection
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt



data = pd.read_csv('training data_including_test_data_corrosion_rate_confirmation.csv')
data.head()
data.tail()
data.shape
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
print ("X_train: ", X_train)
print ("y_train: ", y_train)
print("X_test: ", X_test)
print ("y_test: ", y_test)
###########################
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
################################################

poly_features= PolynomialFeatures(degree=2)
x_poly = poly_features.fit_transform(X)

  
# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)
  
# fit the transformed features to Linear Regression
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
####################################################
# predicting on test data-set
y_pred = poly_model.predict(poly_features.fit_transform(X_test))
print("y_pred:",y_pred)
# evaluating the model on test dataset
pred = y_pred
true = y_test
################ Performance Evaluation #######################################
error = abs(true - pred)
score = abs (r2_score(true, pred))
mse = mean_squared_error(true,pred)
mae = mean_absolute_error(true, pred)
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
plt.title('polynomial regression visualization') 
# show a legend on the plot 
plt.legend() 
# function to show the plot 
plt.show()
  
