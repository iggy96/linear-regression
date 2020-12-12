# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 00:29:26 2020

@author: oseho
"""


# Python program to print current hour, 
# minute, second and microsecond

# importing the datetime class from 
# datetime module
from datetime import datetime

def convert_to_preferred_format(sec):
   sec = sec % (24 * 3600)
   hour = sec // 3600
   sec %= 3600
   min = sec // 60
   sec %= 60
#   print("seconds value in hours:",hour)
#   print("seconds value in minutes:",min)
   return "%02d:%02d:%02d" % (hour, min, sec) 
 
n = 8
print("Superlearner runtime :",convert_to_preferred_format(n))
n = 17
print("Lasso runtime :",convert_to_preferred_format(n))
n = 19
print("SVR Poly Kernel runtime :",convert_to_preferred_format(n))
n = 20
print("Ridge regression runtime :",convert_to_preferred_format(n))
n = 23
print("linear regression runtime :",convert_to_preferred_format(n))
n = 20
print("Extra Gradient Boosting Machine runtime :",convert_to_preferred_format(n))
n = 27
print("Hybrid Metaheuristic Regression Model (SFA-LSSVR) clone runtime :",
      convert_to_preferred_format(n))
