# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 10:03:52 2019

@author: 140524
"""

#https://www.youtube.com/watch?v=8jazNUpO3lQ&list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw&index=2
#Video 2: Linera regression single variable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/1_linear_reg/homeprices.csv')
df
plt.xlabel('area(sq ft)')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red', marker='*')
plt.show()
reg = LinearRegression()
reg.fit(df[['area']],df.price)

reg.predict([[3300]]) 
reg.coef_
reg.intercept_
df
new_df = df.drop(columns='price',axis=1)
new_df
df.price
reg.fit(new_df,df.price)
df1 = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/1_linear_reg/areas.csv')
df1
price = reg.predict(df1)
df1['price'] = price
df1
df1.to_csv('C:\python\prediction.csv')

#Excercise
#Predict canada's per capita income in year 2020. There is an exercise folder 
#here on github at same level as this notebook, download that and you will find 
#canada_per_capita_income.csv file. Using this build a regression model and predict 
#the per capita income fo canadian citizens in year 2020
usecols = ['year','income']
canada = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/1_linear_reg/Exercise/canada_per_capita_income.csv')
canada.columns = usecols
canada.head()
reg.fit(canada[['year']],canada.income)
reg.coef_
reg.intercept_
reg.predict([[2020]])

#  Video 3: Linear Regression - Multivariate

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/2_linear_reg_multivariate/homeprices.csv')
df.bedrooms = df.bedrooms.fillna(df.bedrooms.median())
df
df.columns
reg.fit(df[['area','bedrooms','age']],df.price)
reg.coef_
reg.intercept_
reg.predict([[3000,3,35]])
reg.predict([[2500, 4, 5]])

#Excercise: In exercise folder (same level as this notebook on github) there is 
#hiring.csv. This file contains hiring statics for a firm such as experience of 
#candidate, his written test score and personal interview score. Based on these 3 factors, 
#HR will decide the salary. Given this data, you need to build a machine learning model 
#for HR department that can help them decide salaries for future candidates. Using this 
#predict salaries for following candidates,

from word2number import w2n
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
df=pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/2_linear_reg_multivariate/Exercise/hiring.csv')
df.experience = df.experience.fillna('zero')
df
df.experience = df.experience.apply(w2n.word_to_num)
df
df.columns
df['test_score(out of 10)']=df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].median())
df
df[['test_score(out of 10)']]
reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])
reg.predict([[2,8,7]])

#video 3

#Gradient Descent, Minima, learning rate, and cost function (mean square error)

#y = mx+c
#m -- slope
#c - intercept or coeff

area = [2600,3000,3200,3600,4000]
price = [550000,565000,610000,680000,725000]

# =============================================================================
# https://www.youtube.com/watch?v=vsWrXfO3wWw&list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw&index=4
# 
# based on the area we predict the home prices
# the price share a linear equation with the area by y=mx+c
# in previous linear regression model we found that the coeff is 180618.43
# and the slope to be 135.78
# so the equation will be: price = 135.78*area+180618.43
# with this we try to create the best fit line
# 
# but we can have so many best fit lines. how do we know which of the line is the best fit line.
# one way to do is: we drop any random line and calculate the distance/error of each data point in the 
# train set (y) and the predicted data point (y). collect all the deltas and square them. The reason
# u want to square them is because you dont want any negative distance and dont want the graph to
# be skeweed
# (1/n)* (y-y_predict)^2 ---suming for all data points - this is the mean squared error or cost function
# y_pricted is mx+c or 135.78*area+180618.43
# we can draw so many lines with all combinations of coeff and intercepts because that is very inefficient
# you want to take a efficient way where with less iteration you can find ur answer
# gradient descent is one such algorithm that finds the best fit line for a given training data set
# 
# The way the gradient descent algo works is by plotting a 3D graph with MSE(cost function) Vs intercept and MSE vs
# coeff. The graph will look like a fishnet bowl as shown in the video
# if we break it into 2 graphs MSE vs coeff and MSE vs intercept and try separately to reach:
# the MINIMA which is the point of least MSE and also reduce ur m and c
# 
# in each graph u start at start point in the graph and take mini steps to reach the gloabl minima
# you will have to find the slope of each step by a minimal movelemnt of MSE with a minimal movement
# of c or M following the curvature. at each point u will have to calculate the slope of the tangent
# to the point
# 
# u can use 'learning step' to reach the next point along the curvature
# for calculating the slope u can use derivative - calculas delta MSE(change in Y)/ delta m or c (change in x):
# as x and y shrinks to 0 u will get more accurate slope
# 
# For a graph y = x^2 meaning a small change of x will lead to small change of y which is small change of 
# x^2
# the slope will be delta y/ delta x or delta x^2/delta x = 2x
# for x = 2 the slope will be 4:
#     
# partial derivative - when u have 2 variables x and y then u take a derivative of x
# keeping y = 0 and then take a derivative of y taking x = 0
# 
# for example: f(x,y) = x^2+y^3
# f'(x) = 2x
# f'(y) = 3y^2    
# 
# MSE = 1/n sum (i=1 to n) (yi-(yi-mxi))^2
# 
# derivative MSE with respect to m will be as shown in video
# 
# 
# =============================================================================

    

def gradient_descent(x,y):
# start from declaring  mcurrtent and using b here instead of c currenbt as 0
          m_curr = b_curr = 0 
#starung with 1000          
          iteration = 1000
          #learning_rate is by trial and error - take a small value and then improve it
          learning_rate = 0.01
          for i in range(iteration):
#calculating the predicted value of y
               y_pred = m_curr*x + b_curr
#derivative of m
               md = (-2/len(x))*sum(x*(y-y_pred))
#derivative of b          
               bd = (-2/len(x))*sum(y-y_pred)
# now taking the baby stpes for m_curr and b_curr - which is slope * learning rate
               m_curr = m_curr-learning_rate*md
               b_curr = b_curr-learning_rate*bd
               cost = (1/len(x))*sum([val**2 for val in (y-y_pred)])
               print("m{},b{},cost{},iteration{}".format(m_curr,b_curr,cost,i))
               

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])
gradient_descent(x,y)    

# now we tweak learning_rate and # of iteration to make iteration minimal and increase
#the learning_rate

#with iteration = 10 and learning rate = 0.08

import numpy as np
def gradient_descent(x,y):
# start from declaring  mcurrtent and using b here instead of c currenbt as 0
          m_curr = b_curr = 0 
#starung with 1000          
          iteration = 10000
          #learning_rate is by trial and error - take a small value and then improve it
          learning_rate = 0.08
          for i in range(iteration):
#calculating the predicted value of y
               y_pred = m_curr*x + b_curr
#derivative of m
               md = (-2/len(x))*sum(x*(y-y_pred))
#derivative of b          
               bd = (-2/len(x))*sum(y-y_pred)
# now taking the baby stpes for m_curr and b_curr - which is slope * learning rate
               m_curr = m_curr-learning_rate*md
               b_curr = b_curr-learning_rate*bd
               cost = (1/len(x))*sum([val**2 for val in (y-y_pred)])
               print("m{},b{},cost{},iteration{}".format(m_curr,b_curr,cost,i))
               

x = np.array([1,2,3,4,5])
x
y = np.array([5,7,9,11,13])
gradient_descent(x,y) 

# see the m and b value now stabilized and the cost function is reduced.

gradient_descent(x,y)

#Excercise: 


import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
def gradient_descent(x,y):
          m_curr = b_curr = 0 
          iteration = 1000
          learning_rate = 0.08
          for i in range(iteration):
               y_pred = m_curr*x + b_curr
               cost = (1/len(x))*sum([val**2 for val in (y-y_pred)])
               md = (-2/len(x))*sum(x*(y-y_pred))      
               bd = (-2/len(x))*sum(y-y_pred)
               m_curr = m_curr-learning_rate*md
               b_curr = b_curr-learning_rate*bd
               if math.isclose(cost, cost_previous, rel_tol=1e-20):
                                          break
               cost_prev = cost 
               print("m{},b{},cost{},iteration{}".format(m_curr,b_curr,cost,i))
          
df = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/3_gradient_descent/Exercise/test_scores.csv')
df
x = np.array(df.math)
y = np.array(df.cs)
gradient_descent(x,y)

import random
for x in range(10):
  temp = random.randint(1,101)
  print('temp:',temp)
  print('---')
  prev_temp=temp
  print('prev_temp:',prev_temp)
  result = prev_temp - temp
  print(result)
  print('---')
    
  
  #---
  
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math

def predict_using_sklean():
    df = pd.read_csv("https://raw.githubusercontent.com/codebasics/py/master/ML/3_gradient_descent/Exercise/test_scores.csv")
    r = LinearRegression()
    r.fit(df[['math']],df.cs)
    return r.coef_, r.intercept_

def gradient_descent(x,y):
    m_curr = 0
    b_curr = 0
    iterations = 1000
    n = len(x)
    learning_rate = 0.0002

    cost_previous = 0

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n)*sum([value**2 for value in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        if math.isclose(cost, cost_previous, rel_tol=1e-20):
            break
        cost_previous = cost
        print ("m {}, b {}, cost {}, iteration {}".format(m_curr,b_curr,cost, i))

    return m_curr, b_curr

if __name__ == "__main__":
    df = pd.read_csv("https://raw.githubusercontent.com/codebasics/py/master/ML/3_gradient_descent/Exercise/test_scores.csv")
    x = np.array(df.math)
    y = np.array(df.cs)

    m, b = gradient_descent(x,y)
    print("Using gradient descent function: Coef {} Intercept {}".format(m, b))
    
#Video 6: Dummy Variable and One Hot encoding

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
df = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/5_one_hot_encoding/homeprices.csv')
df    
df1 = pd.get_dummies(df,columns = ['town'],drop_first=True)
df1

# or the long way

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
df = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/5_one_hot_encoding/homeprices.csv')
df    
dum = pd.get_dummies(df.town)
dum
df1 = pd.concat([df,dum],axis = 'columns')
df1.drop(['town','west windsor'],axis='columns',inplace = True)
df1
X = df1.drop('price', axis = 'columns')
y = df1.price
reg = LinearRegression()
reg.fit(X,y)
z = reg.predict(X)
result = pd.DataFrame({'true':y,'predict':z})
result
reg.score(X,y)
reg.predict([[3400,0,0]])

#one hot encoding
#First step is to use label encoder to convert town names into numbers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dfle = df
dfle.town = le.fit_transform(dfle.town)
dfle
X=dfle[['town','area']].values
X
y=dfle['price'].values
y
#Now use one hot encoder to create dummy variables for each of the town
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0])
X = ohe.fit_transform(X).toarray()
X
X = X[:,1:]
X
reg.fit(X,y)
reg.predict([[0,1,3400]])


# Excercise
#At the same level as this notebook on github, there is an Exercise folder that 
#contains carprices.csv. This file has car sell prices for 3 different models. 
#First plot data points on a scatter plot chart to see if linear regression model can be applied. 
#If yes, then build a model that can answer following questions,
#
#1) Predict price of a mercedez benz that is 4 yr old with mileage 45000
#
#2) Predict price of a BMW X5 that is 7 yr old with mileage 86000
#
#3) Tell me the score (accuracy) of your model. (Hint: use LinearRegression().score())

df = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/5_one_hot_encoding/Exercise/carprices.csv')
df
df1 = pd.get_dummies(df,columns = ['Car Model'],drop_first=True)
df1.columns
X = df1.drop('Sell Price($)',axis = 'columns')
X.columns
y= df1['Sell Price($)']
reg.fit(X,y)
reg.predict([[45000,4,0,1]])
reg.predict([[86000,7,1,0]])
reg.score(X,y)

#Using OneHotEncoding
df = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/5_one_hot_encoding/Exercise/carprices.csv')
df
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
dfle = df
dfle['Car Model'] = le.fit_transform(dfle['Car Model'])
dfle
X = dfle[['Car Model','Mileage','Age(yrs)']].values
X
y = dfle[['Sell Price($)']].values
y
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0])
X = ohe.fit_transform(X).toarray()
X
X = X[:,1:]
X
reg.fit(X,y)
reg.predict([[45000,4,0,1]])
reg.predict([[86000,7,1,0]])
reg.score(X,y)

#video 8: Logistic Regression - Binary classification

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
df = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/7_logistic_reg/insurance_data.csv')
df.columns
plt.xlabel('age')
plt.ylabel('bought_insurance')
plt.scatter(df.age,df.bought_insurance,marker = '*', color = 'red')
plt.show()
X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance,train_size=0.9)
log = LogisticRegression()
log.fit(X_train,y_train)
y_pred = log.predict(X_test)
result = pd.DataFrame({'true':y_test,'y_pred':y_pred})
result
print(metrics.accuracy_score(y_test,y_pred))
print('$$$$')
print(log.score(X_test,y_test))

#Excercise

#Download employee retention dataset from here: 
#    https://www.kaggle.com/giripujar/hr-analytics.
#
#Now do some exploratory data analysis to figure out which variables have direct 
#and clear impact on employee retention (i.e. whether they leave the company or 
#                                        continue to work)
#Plot bar charts showing impact of employee salaries on retention
#Plot bar charts showing corelation between department and employee retention
#Now build logistic regression model using variables that were narrowed down in step 1
#Measure the accuracy of the model

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/7_logistic_reg/Exercise/HR_comma_sep.csv')
df.head()
df.columns
df.left
Left = df[df.left==1]
Left.shape
left_low = df[(df.left==1) & (df.salary=='low')]
a = len(left_low)
left_medium = df[(df.left==1) & (df.salary=='medium')]
b = len(left_medium)
left_high = df[(df.left==1) & (df.salary=='high')]
c = len(left_high)
L = [a,b,c]
L
Retained = df[df.left==0]
Retained.shape
R_low = df[(df.left==0) & (df.salary=='low')]
d = len(R_low)
R_medium = df[(df.left==0) & (df.salary=='medium')]
e = len(R_medium)
R_high = df[(df.left==0) & (df.salary=='high')]
f = len(R_high)
R=[d,e,f]
R
S = ['low','medium','high']
index_x = np.arange(len(S))
plt.xlabel('salary')
width = 0.25

plt.ylabel('left and retained')
plt.bar(index_x+width,L,width = width,color='#008fd5',label='Left')
plt.bar(index_x,R,width=width,color='#e5ae38',label='Retained')
plt.xticks(ticks = index_x, labels = S)
plt.show()
plt.tight_layout()
plt.style.use('fivethirtyeight')

Dept = df.Department.unique()
L_D = df.groupby('Department').left.sum()
L_D.index
L_D.values
df1 = df.replace({0:1, 1:0})
R_D = df1.groupby('Department').left.sum()
X = R_D.index
X_tag = np.array(X)
X_tag
y = R_D.values
y
X_index= np.arange(len(X))
X_index
width = 0.2
#plt.xticks(ticks = X_tag)
plt.plot(X_index,y)
plt.bar(X_index, y, width = width, color='#008fd5')
plt.tight_layout()
plt.style.use('fivethirtyeight')
plt.show()

#solution: This is important: see the impact of cross tabulation

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/7_logistic_reg/Exercise/HR_comma_sep.csv')
df.head()
df.columns

#Data exploration and visualization

left = df[df.left==1]
left.shape

Retained = df[df.left==0]
Retained.shape

#Average numbers for all columns

df.groupby('left').mean()

#From above table we can draw following conclusions,

#**Satisfaction Level**: Satisfaction level seems to be relatively low (0.44) in 
#employees leaving the firm vs the retained ones (0.66)
#**Average Monthly Hours**: Average monthly hours are higher in employees leaving the firm (199 vs 207)
#**Promotion Last 5 Years**: Employees who are given promotion are likely to be retained at firm


#impact on salary on employee retention

pd.crosstab(df.salary,df.left).plot(kind = 'bar')

#Department wise employee retention rate

pd.crosstab(df.Department,df.left).plot(kind='bar')

#From above chart there seem to be some impact of department on employee retention 
#but it is not major hence we will ignore department in our analysis

##From the data analysis so far we can conclude that we will use following variables 
#as dependant variables in our model
#**Satisfaction Level**
#**Average Monthly Hours**
#**Promotion Last 5 Years**
#**Salary**

sub_table = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
sub_table

#or

sub_table = df.loc[:,['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
sub_table

sub_table1 = pd.get_dummies(sub_table, columns = ['salary'],drop_first=True)
sub_table1.columns
X = sub_table1.drop('salary_low',axis = 'columns')
X.columns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
y = df.left
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.9)
log = LogisticRegression()
log.fit(X_train,y_train)
y_pred = log.predict(X_test)
y_pred
print(metrics.accuracy_score(y_test,y_pred))
print('$$$$')
print(log.score(X_test,y_test))


# Video 9

# Multiclass logistic regression

#identify handwritten digits

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()
digits
type(digits)
dir(digits)
#each data is an one Dimensional array of numbers. The total array of data[0] 
#represents 1 image. and so on
digits.data[0]
#printing the image
plt.gray()
plt.matshow(digits.images[0])

#printing the first 5 images

for i in range(5):
    plt.matshow(digits.images[i])
#the target gives the identify of the image   
digits.target[0:5]
#each data is an array iof numbers representing 1 hand written image 
digits.data[0:5]
digits.target_names[0:5]
    
#we can use the data and target to built the model

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.2)
len(X_train)

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
logreg.score(X_test,y_test)

#testing with a random sample

plt.matshow(digits.images[67])
digits.target[67]
#now that we know the answer lets see what our model will predict

logreg.predict([digits.data[67]])

# how do i know where exactly the model failed
y_pred = logreg.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm

#ploting confusion matrix

import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

#Excercise

#Use sklearn.datasets iris flower dataset to train your model using logistic regression. 
#You need to figure out accuracy of your model and use that to predict different samples in 
#your test dataset. In iris dataset there are 150 samples containing following features,
#
#Sepal Length
#Sepal Width
#Petal Length
#Petal Width
#Using above 4 features you will clasify a flower in one of the three categories,
#
#Setosa
#Versicolour
#Virginica

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
iris = load_iris()
dir(iris)
iris.data[0]
iris.feature_names[0]
iris.filename[0]
iris.target[0]
iris.target_names[0]
from sklearn.model_selection import train_test_split
X = iris.data
y = iris.target
log = LogisticRegression()
X_train,X_test,y_train,y_test=train_test_split(X,y, random_state=4,test_size = 0.3)
log.fit(X_train,y_train)
y_pred = log.predict(y_test)
y_pred
from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred))
print('$$$$')
print(log.score(X_test,y_test))

#video 10: Decision Tree

import pandas as pd
df=pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/9_decision_tree/salaries.csv')
df
df.columns
inputs = df.drop('salary_more_then_100k', axis = 'columns')
inputs
target = df['salary_more_then_100k']
target
from sklearn.preprocessing import LabelEncoder
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()
inputs['company_n'] = le_company.fit_transform(inputs.company)
inputs['job_n'] = le_company.fit_transform(inputs.job)
inputs['degree_n'] = le_company.fit_transform(inputs.degree)
input_n = inputs.drop(['company','job','degree'],axis = 'columns')
input_n
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(input_n,target)
from sklearn import metrics
model.score(input_n,target)

#Excercise

# =============================================================================
# Build decision tree model to predict survival based on certain parameters
# 
# <img src="titanic.jpg" height=200 width=400/>
# 
# CSV file is available to download at https://github.com/codebasics/py/blob/master/ML/9_decision_tree/Exercise/titanic.csv
# 
# In this file using following columns build a model to predict if person would survive or not,
# Pclass
# Sex
# Age
# Fare
# =============================================================================


url = 'https://raw.githubusercontent.com/codebasics/py/master/ML/9_decision_tree/Exercise/titanic.csv'
titanic = pd.read_csv(url)
titanic.head()
titanic['Sex']
titanic['Sex'] = titanic.Sex.map({'male':1,'female':0})
titanic['Sex']
titanic.head()
titanic.columns
#finding sount of null values in each coluumn
titanic.isnull().sum()
## fill in the missing values for age with the median age
titanic.Age.fillna(titanic.Age.median(),inplace=True)
##fillin the missing value for embarked with the mode
titanic.isnull().sum()

feature_cols = ['Pclass', 'Sex', 'Age', 'Fare']
X = titanic[feature_cols]
y = titanic['Survived']

# fit a classification tree with max_depth=3 on all data
from sklearn.tree import DecisionTreeClassifier
treeClf = DecisionTreeClassifier(max_depth = 3, random_state = 1)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(treeClf, X, y, cv=14, scoring='neg_mean_squared_error')
scores
np.mean(np.sqrt(-scores))
treeClf.fit(X,y)

# compute the feature importances
pd.DataFrame({'feature':feature_cols, 'importance':treeClf.feature_importances_})


#video 11

# SVM - Support vector machine

#Kernel: 3 types of Kernel ---Linear, Polynomial, RBF
#Linear  - when data is lineraly separable, we use linear kernel
#Polynomial - when data is non linearly separable and can be classified using a
#curve, we use polynomial
#RBF - when data is non linearly separable but canNOT be classified using a curve, we 
#use RBF. This results in hyperplane creation
#
#Gamma - Gamma is the parameter in gaussian kernel to handle NON Linear separation and RBF
#when u can't separate the data in 2D and you have to create a hyperplace, Gamma is the
#parameter that control the measure of the peaks of the hyper plane separating the data
#a low gamma gives u a pointed bump/peak in the hyperplane boundary 
#a high gamma gives u a softer broader bump/peak
#so a low gamma will give u a high bias and low variance while
#a high gamma will give u a low bias and high variance
#this is a bias variance trade off u have to optimize.
#
#Regularization C: C deteremines the width of the margin separating the data
#higher the C value, lower the margin. 
#A high C gives u a low bias and high variance
#A low C will give u a low bias and high variance
#this is a bias variance trade off u have to optimize.


import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()

dir(iris)
iris.feature_names

# converting the sklearn data set into dataframe

df = pd.DataFrame(iris.data, columns = iris.feature_names)
df.head()

# adding the target column into the df dataframe
df['target'] = iris.target
df
df[df.target==0]
# the above shows that rows 1 to 49 are for target 0
df[df.target==1]
# the above shows that rows 50 to 99 are for target 1
df[df.target==2]
# the above shows that rows 100 to 149 are for target 1


# Additng flower names to target

# way one

df['flower'] = df.target.map({0:'setosa', 1: 'versicolor', 2: 'virginica'})
df

# or

# way 2
iris.target_names
df['flower'] = df.target.apply(lambda x: iris.target_names[x])
df

df0=df[:50]
df1=df[50:100]
df2=df[100:]


# plotting data to see how to separate into 3 classifications

# classifying through sepal length and width

import matplotlib.pyplot as plt
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="blue",marker='.')
plt.scatter(df2['sepal length (cm)'], df2['sepal width (cm)'],color="red",marker='x')

# similarly classifying through petal length and width

import matplotlib.pyplot as plt
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],color="green",marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color="blue",marker='.')
plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'],color="red",marker='x')

# Train SVM

from sklearn.model_selection import train_test_split
X = df.drop(['target','flower'], axis = 1)
X
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
len(X_train)
len(X_test)

from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
model.score(X_test, y_test)
model.predict([[4.8,3.0,1.5,0.3]])

#Tuning parameters
#Regularization

model_C = SVC(C=1)
model_C.fit(X_train, y_train)
model_C.score(X_test, y_test)

#Gamma

model_g = SVC(gamma=10)
model_g.fit(X_train, y_train)
model_g.score(X_test, y_test)

#kernel

    model_linear_kernal = SVC(kernel='linear')
    model_linear_kernal.fit(X_train, y_train)
    model_linear_kernal.score(X_test, y_test)

# Excercse
    
#Train SVM classifier using sklearn digits dataset (i.e. from sklearn.datasets 
#import load_digits) and then,
#
#Measure accuracy of your model using different kernels such as rbf and linear.
#Tune your model further using regularization and gamma parameters and try to come up with highest accurancy score
#Use 80% of samples as training data size  
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()
digits
type(digits)
dir(digits)
df = pd.DataFrame(digits.data)
df.head()
df['target'] = digits.target
df


X = digits.data
y = digits.target

param_dist = dict(C = np.arange( 1, 100+1, 1 ).tolist(), kernel = ['linear', 'rbf']  , gamma = np.arange( 0.0, 10.0+0.0, 0.1 ).tolist())
param_dist


from sklearn.svm import SVC
model = SVC()
from sklearn.model_selection import RandomizedSearchCV
rand = RandomizedSearchCV(model,param_dist,cv=10,scoring='accuracy',n_iter=10,random_state=5,return_train_score=False)
rand.fit(X,y)
results = pd.DataFrame(rand.cv_results_)
results.head()
results.columns
results[['mean_test_score', 'std_test_score', 'params']]
#see even though it searched just 10 combinations it manaded to find a best score
#like gridseacrhcv
print(rand.best_score_)
print(rand.best_params_)
print(rand.best_estimator_)


# Video 12

# Random Forest

# n_estimator is the hyper parameter here by which u can control the number of 
#decision trees in the forest to optimize output

    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()
digits
type(digits)
dir(digits)
df = pd.DataFrame(digits.data)
df.head()
df['target'] = digits.target
df
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df.drop('target',axis=1),digits.target,test_size=0.2)
len(X_train)
len(X_test)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
model.score(X_test,y_test)

from sklearn import metrics
cm = metrics.confusion_matrix(y_test,y_pred)
cm

# plotting confusion matrix by heatmap
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('True')


# Excercise 

import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()

dir(iris)
iris.feature_names
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df.head()

df['target'] = iris.target
df

# Additng flower names to target

# way one

df['flower'] = df.target.map({0:'setosa', 1: 'versicolor', 2: 'virginica'})
df

# or

# way 2
iris.target_names
df['flower'] = df.target.apply(lambda x: iris.target_names[x])
df

from sklearn.model_selection import train_test_split
X = df.drop(['target','flower'], axis = 1)
X
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
len(X_train)
len(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
Est_range = range(1,100)
score = []
for i in Est_range:
    model=RandomForestClassifier(n_estimators = i)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    score.append(metrics.accuracy_score(y_test,y_pred))
score    
import matplotlib.pyplot as plt
plt.plot(Est_range,score)
plt.xlabel('n_estimator_value')
plt.ylabel('score')
plt.show()    
# the best estimator value is at 5
model=RandomForestClassifier(n_estimators = 5)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred))

cm = metrics.confusion_matrix(y_test,y_pred)
cm

# plotting confusion matrix by heatmap
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('True')

#Video 13

# K Mean

# Un supervised Learning - means the response is not set or categorized yet
# the way to apprach is to cluster the data set based on features
# n_clusters is the main attribute here that you want to optimize
# Elbow curve - the point on the elbow gives u the optimized value of n_cluster
# use MaxMinScaler to scale data for better clustering

from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
df = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/13_kmeans/income.csv')
df   

plt.scatter(df['Age'], df['Income($)'])
plt.xlabel('Age')
plt.xlabel('Income')


km = KMeans(n_clusters=3)
y_pred = km.fit_predict(df[['Age','Income($)']])
y_pred

df['cluster'] = y_pred
df.drop('Name', axis = 1, inplace = True)

km.cluster_centers_

df0 = df[df.cluster==0]
df1 = df[df.cluster==1]
df2 = df[df.cluster==2]

plt.scatter(df0.Age, df0['Income($)'], color = 'red')
plt.scatter(df1.Age, df1['Income($)'], color = 'blue')
plt.scatter(df1.Age, df1['Income($)'], color = 'green')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color = 'purple',marker = '+')
plt.xlabel('Age')
plt.xlabel('Income')


# see the cluster blue and red is not perfect. This is because the data is not scaled
#the income is in 1000s while the age in 10s
# so now scaling the data

scaler = MinMaxScaler()
df['Income($)'] = scaler.fit_transform(df[['Income($)']])
df['Income($)']

df['Age'] = scaler.fit_transform(df[['Age']])
df['Age']
sse = []
n_range=range(1,10)
for i in n_range:
    km = KMeans(n_clusters=i)
    km.fit_predict(df[['Age','Income($)']])
    sse.append(km.inertia_)
    
plt.plot(n_range,sse)
plt.xlabel('n value')
plt.ylabel('sse')   

# the elbow value observed is 3 
km = KMeans(n_clusters=3)
y_pred = km.fit_predict(df[['Age','Income($)']])
df['cluster1'] = y_pred
df
km.cluster_centers_

df0 = df[df.cluster1==0]
df1 = df[df.cluster1==1]
df2 = df[df.cluster1==2]

plt.scatter(df0.Age, df0['Income($)'], color = 'red')
plt.scatter(df1.Age, df1['Income($)'], color = 'blue')
plt.scatter(df1.Age, df1['Income($)'], color = 'green')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color = 'purple',marker = '+')
plt.xlabel('Age')
plt.xlabel('Income')



# Excercise
#
#Use iris flower dataset from sklearn library and try to form clusters of flowers using petal width and length features. Drop other two features for simplicity.
#Figure out if any preprocessing such as scaling would help here
#Draw elbow plot and from that figure out optimal value of k


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
iris = load_iris()
dir(iris)
iris.data[0]


import pandas as pd
X=pd.DataFrame(iris.data)
X

sse = []
n_range=range(1,20)
for i in n_range:
    km = KMeans(n_clusters=i)
    km.fit_predict(X)
    sse.append(km.inertia_)
    
plt.plot(n_range,sse)
plt.xlabel('n value')
plt.ylabel('sse') 

# n_cluster value of 3
km = KMeans(n_clusters=3)
y_pred = km.fit_predict(X)

km.labels_

from sklearn import metrics
print(metrics.silhouette_score(X, km.labels_))
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

import numpy as np
colors = np.array(['red', 'green', 'blue', 'yellow'])
colors
pd.scatter_matrix(X, c=colors[beer.cluster], figsize=(10,10), s=100)


# scaling the data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
km = KMeans(n_clusters=3, random_state=1)
km.fit(X_scaled)
X['cluster'] = km.labels_
X.sort_values('cluster')
km.cluster_centers_
# see now it is different bcoz the data is scaled
beer.groupby('cluster').mean()
# scatter plot matrix of new cluster assignments (0=red, 1=green, 2=blue)
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

import numpy as np
colors = np.array(['red', 'green', 'blue', 'yellow'])
colors
pd.scatter_matrix(X, c=colors[beer.cluster], figsize=(10,10), s=100)



from sklearn import metrics
metrics.silhouette_score(X_scaled, km.labels_)

# calculate SC for K=2 through K=19
k_range = range(2, 20)
scores = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=1)
    km.fit(X_scaled)
    scores.append(metrics.silhouette_score(X_scaled, km.labels_))
    
scores 
# plot the results
plt.plot(k_range, scores)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Coefficient')
plt.grid(True)

km = KMeans(n_clusters=3, random_state=1)
km.fit(X_scaled)
beer['cluster'] = km.labels_
beer.sort_values('cluster')

km.cluster_centers_  
km.inertia_