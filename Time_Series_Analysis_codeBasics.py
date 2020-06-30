# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 22:09:51 2019

@author: 140524
"""

#Video 1: Code Basics
#DateTimeIndex
#Resampling
#https://www.youtube.com/watch?v=r0s4slGHwzE&list=PLeo1K3hjS3uuASpe-1LjfG5f14Bnozjwy&index=14

#DateTimeIndex
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/pandas/14_ts_datetimeindex/aapl.csv')
df.head()

#see the Data type of the date column is a string

type(df.Date[1])
# or
type(df.Date[0])
# or
type(df.Date)

# if we want to change the type of the Date column from type string to type Date
# then we need to add a porameter in the read_csv called - parse_date


df = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/pandas/14_ts_datetimeindex/aapl.csv',parse_dates=['Date'])
df.head()
type(df.Date[1])

# if you want the Date column as the index then

df = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/pandas/14_ts_datetimeindex/aapl.csv',parse_dates=['Date'], index_col='Date')
df.head()

# see now the index is of type datetime index

df.index

# there are many benefits of changing the index to datetime for example we want the Jan 2017 data
#it will be easy

df['2017-01']

#finding average stock price of Apple for Jan

df['2017-01'].Close.mean()

# lets check the price of a single day - 3rd Jan 2017

df['2017-01-03']

# checking the stock prices of a date range

df['2017-01-10':'2017-01-1']

#Resampling - we can find the monthly mean - M for monthly frequency

df.Close.resample('M').mean()

#plotting the above

import matplotlib.pyplot as plt
df.Close.resample('M').mean().plot()

#weekly frequency

df.Close.resample('W').mean().plot()

#frequency chart of pandas:
#https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html

#or Quarterly frequency in bar charts

df.Close.resample('Q').mean().plot(kind = 'bar')

# or plotting the closing the price chart

df.Close.plot()

#====================================================================================
#Video 2 Date Range
#https://www.youtube.com/watch?v=A9c7hGXQ5A8&list=PLeo1K3hjS3uuASpe-1LjfG5f14Bnozjwy&index=15

#if the date range is missing in the data set then dat range fucntion will help add the dates 
# to the data set even taking care of the weekends

df_1 = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/9c5b7d5d4b242d544700c4e36ccb1b6d888619aa/pandas/15_ts_date_range/aapl_no_dates.csv')
df_1.head()

# we have to insert dates into this dataframe, for this we have to use date range function
#insert start date, end date and the freequency is B for business days

df2 = pd.date_range(start='6/1/2017', end='6/30/2017', freq = 'B')
df2

df_1.set_index(df2,inplace=True)
df_1
#plotting the closing price

df_1.Close.plot()

#if we want the weekend values as well, and the weekend value will be the friday value
#carry forwarded into weekend, then we use the asfreq method and freq parameter as D

df_1.asfreq('D',method='pad')

#if used for getting the weekly price

df_1.asfreq('W',method='pad')

#or hourly price

df_1.asfreq('H',method='pad')

#tocarry over data from previous date/ hour use asfreq

#if we dont know the end date, we know the start date only but we know the number of days

df2 = pd.date_range(start='1/1/2017', periods = 72, freq='B')
df2

#or may be hourly freq

df3 = pd.date_range(start='1/1/2017', periods = 72, freq='H')
df3

#lets say we generate random numbers and we have the date_range we generated.
#together we can build a series or a dataframe

import numpy as np
ts = np.random.randint(1,100,len(df2))
ts

ts1=pd.Series(np.random.randint(1,100,len(df2)), index=df2)
ts1

#this us very use to generate a random sample

#imp point: the date range will not handle holiday

#===========================================================================
#Video 3: handling holidays in panda time series

#https://www.youtube.com/watch?v=Fo0IMzfcnQE&list=PLeo1K3hjS3uuASpe-1LjfG5f14Bnozjwy&index=16

#generate a custom holiday calendar and create a date time index out of it

df_1 = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/9c5b7d5d4b242d544700c4e36ccb1b6d888619aa/pandas/15_ts_date_range/aapl_no_dates.csv')
df_1.head()

# lets say these are the July stock prices. Using frequency B will exclude weekends but
#will not exlcude the core holidays such as July 4rth

df2 = pd.date_range(start='7/1/2017', end='8/1/2017', freq = 'B')
df2

df_1.set_index(df2,inplace=True)
df_1

# how to also exlcude July 4th

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
cbd = CustomBusinessDay(calendar = USFederalHolidayCalendar())
cbd
df2 = pd.date_range(start='7/1/2017', end='8/2/2017', freq = cbd)
df2

df_1.set_index(df2,inplace=True)
df_1

#defining a custom holiday

from pandas.tseries.holiday import AbstractHolidayCalendar,nearest_workday, Holiday

class myBirthDayCalendar(AbstractHolidayCalendar):
    rules = [
            Holiday('My Birthday', month=11, day=30)]
    
myc = CustomBusinessDay(calendar = myBirthDayCalendar())
myc

pd.date_range(start='11/1/2018',end='11/30/2018', freq = myc)

#in 2019 my bday comes on saturday so i will observe on the nearest work day
# see here friday 11/29 will be a holiday

from pandas.tseries.holiday import AbstractHolidayCalendar,nearest_workday, Holiday

class myBirthDayCalendar(AbstractHolidayCalendar):
    rules = [
            Holiday('My Birthday', month=11, day=30, observance = nearest_workday)]
    
myc = CustomBusinessDay(calendar = myBirthDayCalendar())
myc

pd.date_range(start='11/1/2019',end='11/30/2019', freq = myc)


# for example in Egypt the working days of a week are sunday to thurs, fri and sat
#are weekends  ..so how to custome fit a calendar?

egypt=  CustomBusinessDay(weekmask='Sun Mon Tue Wed Thu') 
pd.date_range(start='11/1/2019',end='11/30/2019', freq = egypt)

# we can also supply a holiday argument in the egypt calendar

egypt=  CustomBusinessDay(weekmask='Sun Mon Tue Wed Thu', holidays = ['2019-11-26']) 
pd.date_range(start='11/1/2019',end='11/30/2019', freq = egypt)

#================================================

#video 4
#https://www.youtube.com/watch?v=igWjq3jtLYI&list=PLeo1K3hjS3uuASpe-1LjfG5f14Bnozjwy&index=17

#To_datetime - addresses the lack of uniformilty in datetime format
#see the different date time formats
import pandas as pd
dates = ['2017-01-05', 'Jan 5, 2017', '01/05/2017', '2017.01.05', '2017/01/05','20170105']
# see below it convert the diff formats into a uniform format
pd.to_datetime(dates)

# it can also handle the time
dt = ['2017-01-05 2:30:00 PM', 'Jan 5, 2017 14:30:00', '01/05/2016', '2017.01.05', '2017/01/05','20170105']
pd.to_datetime(dt)

#US: mm/dd/yyyy



#Europe: dd/mm/yyyy
# to handle europe format - pass dayfirst argumet
pd.to_datetime('5/1/2016', dayfirst=True)
#or
pd.to_datetime('5-1-2016', dayfirst=True)

#or if the delimiter is custom

pd.to_datetime('5$1$2016', format = '%d$%m$%Y')

#if we pass garbage values in the string func such as abc for date time it
#will raise an exception . however we can ignore the all errors by ---

pd.to_datetime(['2017-01-05', 'Jan 6, 2017', 'abc'], errors='ignore')

# or just this error by

pd.to_datetime(['2017-01-05', 'Jan 6, 2017', 'abc'], errors='coerce')

#Epoch: https://www.epochconverter.com/
#Epoch or Unix time means number of seconds that have passed since Jan 1, 1970 00:00:00 UTC time
#unix time unit is ns - nano second, we have to convert it to sec
current_epoch = 1501324478
pd.to_datetime(current_epoch, unit='s')

# to convert into a date time index we will have to pass in as an array
current_epoch = 1501324478
t = pd.to_datetime([current_epoch], unit='s')
t

# to convert it back to epoch
t.view('int64')

#video 5: Time span, Period and PeriodIndex
#https://www.youtube.com/watch?v=3l9YOS4y24Y&list=PLeo1K3hjS3uuASpe-1LjfG5f14Bnozjwy&index=18

import pandas as pd
y=pd.Period('2016')
y
#A DEC means Annual -ending in december
# propertries of period such as start ime, end time, leap year or not etc...

dir(y)

y.start_time
y.end_time
#creating monthly time period

m = pd.Period('2011-1',freq = 'M')
m
m.start_time
m.end_time
# you can perform arithmatic ops on period objects

m+1 # gives me the next month

#daily time period

d = pd.Period('2017-2-28', freq = 'D')
d
# goes to next month
d+1

# hourly time period
h = pd.Period('2017-2-28 23:00:00', freq = 'H')
h
h.start_time
h.end_time
h+1
#or the same thing can be achieved by

h +pd.offsets.Hour(1)
# or
h +pd.offsets.Hour(5)

# or 

h-6

# quarterly time period

q = pd.Period('2017Q1')
q
q.start_time

q.end_time

#most companies has calendatr year from Jan to Dec, few companies such as walmart
# fiscal year is from Feb to Jan

# how to process the data
# we have to mention that the last querter ends in Jan not dec

q = pd.Period('2017Q1', freq = 'Q-JAN')
q
q.start_time
q.asfreq('M', how = 'start')

q1 = pd.Period('2018Q2',freq='Q-JAN')
q1
q1-q

# creating periodindex

idx = pd.period_range('2011','2017',freq = 'Q')
idx

# for walmart


idx = pd.period_range('2011','2017',freq = 'Q-JAN')
idx

idx[0].start_time
idx[0].end_time

#or if we dont know end date

idx = pd.period_range('2011',periods=10,freq = 'Q-JAN')
idx

# creating a data set with periods

import numpy as np
ps = pd.Series(np.random.randn(len(idx)),idx)
ps
ps.index
ps['2011']
ps['2011':'2013']

# converting to date_time index
pst = ps.to_timestamp()
pst
pst.index

#converting date_time_index to period index

pst.to_period()


# converting teh format, adding 2 new columns, and transposing the walmart financial
#table

import pandas as pd
wmt = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/pandas/18_ts_period/wmt.csv', index_col = 'Line Item')
wmt
# see doing a direct T doesnt give us the appropriate result
wmt1 = wmt.T
wmt1
type(wmt1.index[0])
# converting str to period index

wmt1.index = pd.PeriodIndex(wmt1.index,freq = 'Q-JAN')
wmt1.index

wmt1['start_date'] = wmt1.index.map(lambda x:x.start_time)
wmt1['end_date'] = wmt1.index.map(lambda x:x.end_time)
wmt1

# or =============================================================================
# wmt1['start_date'] = [x.start_time for x in wmt1.index]
# wmt1['start_date']
# wmt1
# wmt1['end_date'] = [x.end_time for x in wmt1.index]
# wmt1
# 
# =============================================================================

#video 5
#https://www.youtube.com/watch?v=9IW2GIJajLs&list=PLeo1K3hjS3uvMADnFjV1yg6E5nVU4kOob&index=6
#TimeZone Handling

import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/pandas/19_ts_timezone/msft.csv', header = 1,parse_dates=['Date Time'],index_col = 'Date Time')
df

df.index
# =============================================================================
# 
# Two types of datetimes in python
# Naive (no timezone awareness)
# Timezone aware datetime
# Convert naive DatetimeIndex to timezone aware DatetimeIndex using tz_localize
# =============================================================================

df =df.tz_localize(tz = 'US/Eastern')
df
df.index

#Convert to Berlin time using tz_convert

df = df.tz_convert('Europe/Berlin')
df

from pytz import all_timezones
print (all_timezones)
print(len(all_timezones))

#Convert to kolkata time

df = df.tz_convert('Asia/Calcutta')
df
df.index

#time zones in date range function
#using tz argument
rng = pd.date_range(start='1/1/2017', periods=10,freq='H', tz='Europe/London')
rng

# there are 2 types of time zones 1) pytz and date_util
#the previous all examples are used by pytz timezones
# the below is how to use dateutil timezone

rng = pd.date_range(start='1/1/2017', periods=10,freq='H', tz='dateutil/Europe/London')
rng
#date util will use all the timezones in the OS and the pytz has a predefined list

rng = pd.date_range(start='2017-08-22 09:00:00', periods=10,freq='30min', tz='dateutil/Europe/London')
s = pd.Series(range(10),index=rng)
s

#===================================================================
#video 6
#https://www.youtube.com/watch?v=0lsmdNLNorY&list=PLeo1K3hjS3uvMADnFjV1yg6E5nVU4kOob&index=7
#Shifting and Lagging

import pandas as pd
df=pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/pandas/20_shift_lag/fb.csv',parse_dates = ['Date'],index_col='Date')
df

#shifting the values by 1 day

df.shift(1).plot()
df.shift(2).plot()

# see the graphs it shifted to the right

# we can also shift back

df.shift(-1)
# you can shift method on dataframe as well time series
#one common use of shiting method is to calculate percentage change in 1 day prices

df['prev_day_price'] = df.shift(1)
df

df['1 day price change'] = df['Price'] - df['prev_day_price']
df

# % diference in 5 days
df['5 day % return'] = (df['Price'] - df['Price'].shift(5))*100/df['Price'].shift(5)
df


#now we want to keep the data points intact and shift the dates

df=df[['Price']]
df
df.index

df.index= pd.date_range(start='2017-08-15',periods = 10, freq = 'B')
df.index

#once u have set a freq we can use the tshoft functions to adjust the dates and not the 
#data points

df.tshift(1)
df.shift(-1)