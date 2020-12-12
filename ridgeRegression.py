# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 11:52:26 2020

@author: oseho
"""



import time
from time import strftime
from datetime import datetime 
from time import gmtime

def start_time_():    
    #import time
    start_time = time.time()
    return(start_time)

def end_time_():
    #import time
    end_time = time.time()
    return(end_time)

def Execution_time(start_time_,end_time_):
   #import time
   #from time import strftime
   #from datetime import datetime 
   #from time import gmtime
   return(strftime("%H:%M:%S",gmtime(int('{:.0f}'.format(float(str((end_time-start_time))))))))

start_time = start_time_()
###############################################################################

#Importing the libraries
from defined_libraries import* 
from feature_set import*

# build ridge model
alphas = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1,0.5, 1]
for a in alphas:
 model = Ridge(alpha=a, normalize=True).fit(X_train,y_train) 
 score = model.score(X, y)
 pred_y = model.predict(X)
 mse = mean_squared_error(y, pred_y) 
ridge_mod=(Ridge(alpha=0.0006, normalize=True)).fit(X_train,y_train)
y_pred = ridge_mod.predict(X_test)

################ Performance Evaluation #######################################
def rmse():
	return sqrt(mean_squared_error(y_test, y_pred))
def mse():
    return (mean_squared_error(y_test,y_pred))
def R2():
    return abs (r2_score(y_test, y_pred))

print('RMSE %.3f' % (rmse()))
print('MSE %.3f' % (mse()))
print('R2 %.3f' % (R2()))
error = abs((y_test - y_pred)/y_pred)
percentage_error = (error*100)

################ visualization #####################################
l = list(range(5)) #index numbers for x axis
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
plt.title('Ridge Regression') 
plt.show()
### Saving result in csv file
d = {'y_test':y_test, 'y_pred':y_pred,'error':error,'percentage error':percentage_error}
prediction = pd.DataFrame(d, columns=None).to_csv('ridge regression prediction.csv')

##############################################################################################
[i for i in range(0,100000000)]
############################################################################## your code here #
#Importing the libraries
from defined_libraries import* 
from feature_set import*

# build ridge model
alphas = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1,0.5, 1]
for a in alphas:
 model = Ridge(alpha=a, normalize=True).fit(X_train,y_train) 
 score = model.score(X, y)
 pred_y = model.predict(X)
 mse = mean_squared_error(y, pred_y) 
 print("Alpha:{0:.6f}, R2:{1:.3f}, MSE:{2:.2f}, RMSE:{3:.2f}"
    .format(a, score, mse, np.sqrt(mse)))

ridge_mod=(Ridge(alpha=0.0006, normalize=True)).fit(X_train,y_train)
y_pred = ridge_mod.predict(X_test)

################ Performance Evaluation #######################################
def rmse():
	return sqrt(mean_squared_error(y_test, y_pred))
def mse():
    return (mean_squared_error(y_test,y_pred))
def R2():
    return abs (r2_score(y_test, y_pred))

print('RMSE %.3f' % (rmse()))
print('MSE %.3f' % (mse()))
print('R2 %.3f' % (R2()))
error = abs((y_test - y_pred)/y_pred)
percentage_error = (error*100)

################ visualization #####################################
l = list(range(5)) #index numbers for x axis
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
plt.title('Ridge Regression') 
plt.show()
### Saving result in csv file
d = {'y_test':y_test, 'y_pred':y_pred,'error':error,'percentage error':percentage_error}
prediction = pd.DataFrame(d, columns=None).to_csv('ridge regression prediction.csv')

##############################################################################################
end_time = end_time_()
print("Execution_time is :", Execution_time(start_time,end_time))