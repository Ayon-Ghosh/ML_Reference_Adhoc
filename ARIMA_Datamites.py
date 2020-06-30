# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:30:54 2019

@author: 140524
"""

#ARIMA Datamites
#https://www.youtube.com/watch?v=D9y6dcy0xK8

import pandas as pd
import matplotlib.pyplot as plt
car_sales = pd.read_csv('C:\\Users\\140524\\Desktop\\working_edu\\codes_in_class_13\\sales-cars.csv',parse_dates=['Month'],index_col='Month')
car_sales.head()

car_sales.plot()

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(car_sales)

# trying to make the data stationary
# I of ARIMA - Integrated order of 1 - difference of 1 

sales_diff = car_sales - car_sales.shift(1)
sales_diff.head()
sales_diff.dropna(axis = 0,inplace=True)
sales_diff

plot_acf(sales_diff)
sales_diff.plot()

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(sales_diff, test_size=0.2)
train_data.values
test_data.values
#AR Autoregressive Model

from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
model_ar = AR(train_data.values)
model_ar_fit = model_ar.fit()
predictions = model_ar_fit.predict(test_data.values)
predictions
