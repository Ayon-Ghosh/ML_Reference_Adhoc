# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 18:35:36 2019

@author: 140524
"""

#https://www.youtube.com/watch?v=elojMnjn4kk&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A

# KNN and Logistic regression
#video  =3
#getting started with Iris dataset
#the convention of sklearn is to import indivitual objects, classes, function
#rather than importing the whole module
#imports load_iris function from sklearn.datasets module
from sklearn.datasets import load_iris
#because iris dataset is so popular as a toy dataset it is built into scikitlearn
#saves 'bunch' object containing iris dataset and attributes into iris bar
#we run the load_iris() and save the return value in a object called iris
iris = load_iris()
iris
#iris.dtype
type(iris)
#one of the attributes of this object iris is data which we r printing
print(iris.data)
print(type(iris.data))
#each row is known as observation and each columnis called as feature
# feature also knows as -- predrictor, attribute, independant variable, input,
#regressor, covariate
print(iris.feature_names)
# see classification below - classification is a supervised learning where the 
#response is categorical
# regression is a supervised learning where outcome is contininous and ordered
# each value we are predicting in response also known as target, outcome, label, dependant variable
type(iris.target)
print(iris.target)
print(iris.target_names)

# Features and response are separate objects
#Features and response should be numeric maning only numbers in those 2 objcts
#this is why instead of setosa, vitusa etc...its stores as 0.1.2
#Features and response should be numpy arrays
#Features and response should have specific shapes
# the feature object must be 2 dimensions numpy array  - records/ observations/rows and attributes/ column/ features
# the response object must be of 1 dimesion numpy array

# there should be 1 response corresponding to each obseravtion

# Note that the X is capitalised becaue it represents a matrix while y is small because it represents a vector
X = iris.data
y = iris.target

#video - 4

# training a model
# 4 features (sepal length, sepal width, petal length, petal width)
# response variable as iris species
#classification problem as response is categorical

# k nearest neighbors classification
# Knearest KNN classification steps
#pick the value of K such as 5
# The model searches search for the K (such as 5) observations in the training 
#dataset that are nearest to the measurements 
#of the uknown iris - that is the model calculate the numerical distance between the unknown
#iris and each of the 150 known irises and selects the K (5) knowns irises that are closest
# in distance to the unknown iris - eucldian distance is considered
#use the most popular response from the knearest neighbors as the preducted response value of the
#unknown iris - the response of the k (5) polular response are tallied and whichever closest
# is given as the predicted value for the unknown iris

#step 1 - import the class you plan to use -  in this case we import k neighbors classfier
from sklearn.neighbors import KNeighborsClassifier

#step 2 - instantiate the estimator
#estimator is scikit learns term for model to estimate unknown quantities
# instantiate means making instance of kneighbor classfier class

knn = KNeighborsClassifier(n_neighbors=1)

#Note - Name of the object doesnt matter
#Can specify tuning parameter (aka: hyperparameters during the step)
#all parameters not specified are set to default--here we set n_neighbors parameter to1
# meaning it looks for 1 nearest neighbor . n neighbor parameter is known as 
#tuning or hyper parameter - the other parameters are set to default values

#tHE DEFAULT PARAMETERS BELOW WHEN U PRINT
#NeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#                     metric_params=None, n_jobs=None, n_neighbors=1, p=2,
#                     weights='uniform')
print(knn)

#step4 - fit the model with data )aka: model training
X = iris.data
y = iris.target

knn.fit(X,y)
# predicting with a unknown data - returns a numpy array
# new observations are called out of sample observations
#below doesnt work
#knn.predict([3,5,4,2])
#to make it work
# option 1: pass the data as a nested list, which will be interpreted as having shape (1, 4)
knn.predict([[3, 5, 4, 2]])
# option 2: explicitly change the shape to be (1, 4)
import numpy as np
knn.predict(np.reshape([3, 5, 4, 2], (1, 4)))
# option 3: explicitly change the first dimension to be 1, let NumPy infer that the second dimension should be 4
knn.predict(np.reshape([3, 5, 4, 2], (1, -1)))
# can predict multiple observations
X_new = [[3,5,4,2],[5,4,3,2]]
knn.predict(X_new)
# array(2,1) means the roediction for ist unknowsn observation is 2 and second 1

#using different value of K
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
knn.predict(X_new)
# see the preiction here changes to array(1,1) which is 1 - setosa for each onservation

#Logistic Regression is another classification model despite the name - regression

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X,y)
logreg.predict(X_new)
# predict a value of 2 for ist unknown and 0 for second

# see different models give different results - so how to estimate and chose the 
#best model that gives the best prediction
#see next video

#  video = 5

# how do i choose the best model to use from supervised learning
#how do i choose the best tuning parameters
#how do i estinate the likely performance of any model from an out of sample data

#used KNN with K-1 or 5 and logistic repression - this is called model evaluation
#procedure where we can choose the likely performance of 3 models and choose the best 
#model

#evaluation procedure 1 - Train and test (though it doesnt have a name)
#train the model with the entire data set
#test the model with the same dataset and evaluate how well we did by comparing 
#the predicted response value with the true response value

# bychecking the predicted response and conparing it with true response in the stored 
#dataset we know how well our model is performating

#evaluating first logistic regression

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
X = iris.data
y = iris.target
logreg.fit(X,y)
#testing - predicting with training data only

y_pred = logreg.predict(X)
len(y_pred)
# now evalution
#use classification accuracy
#this is a common evaluation metric for classicifcation problems
#it measures the proportion of correct predictions (comparing predicted with true response
#because remember true response is part of the training data set u are using to test as well)

from sklearn import metrics
#known as training accuracy when u train and test the model on the same data
print(metrics.accuracy_score(y,y_pred))
# this shows 96% was correct

# KNN = 5 - measuring the accuracy

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y,y_pred))
#this has a higher accuracy - 0.9666667

# KNN=1
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X,y)
y_pred = knn.predict(X)
y_pred
print(metrics.accuracy_score(y,y_pred))
# highest accurcy of 1
# so we will conclude that KNN with KNN=1 would be the best model for this conclusion

# Drawback on KNN=1 with K = 1 it provides the response value of the nearest neighbor
#now testing with training data means it memorised the training data and is providing the
#exact feature match
#this is the drawback when u use training data to test

# problem with this evaluation model - train and test data
#see the goal is to estimate likely performance of a out of sample data
#But maximising training accuracy only rewards with highly complex models that 
#wont necessarily generalize to future cases (meaning accurately predict out of sample data)
#creating unneccesarily complex model is known as overfit - models over fit the training data
#reads the noise of the data pretty accurately but doesnt read the signal from the data
#in the case of KNN = 1 it creats a high complex model/ over fits that follows the noise
#in the date \\\diagram shown on |9:25min of the video

#concept of DESICISION BOUNDARY --- |10:37min

# second evaluation model - train test split/test set apprach/ validation set approach

#split the data into training set and testing set
#train the model on the training set and test the model on the testing set and evaluate how well we did

# note the shape of X and y
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
X.shape
y.shape
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
# splitting the data in 60-40
# if u dont assign a intiger to random state, every time the training data will be 
#different which wont give u a good result. Assigning any value to random state
#(1-5-100 - anyval;ue etc..what is the value doesnt matter) will train the machine with the 
#same dataset

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4, random_state=7)
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
#testing - predicting with training data only

y_pred = logreg.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred))

# testing accuracy is 91%

# testing accuracy with KNN = 5; GIVES out to be 93%

from sklearn.neighbors import KNeighborsClassifier
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4, random_state=7)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred))

# testing accuracy with KNN = 1; GIVES out to be 95%

from sklearn.neighbors import KNeighborsClassifier
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4, random_state=7)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred))


##### *** finding a even better value of K
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
k_range=range(1,26)
score = []
for k in k_range:
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4, random_state=4)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    score.append(metrics.accuracy_score(y_test,y_pred))
#thelist prints all the value of accuracy for each K
score    
dict(zip(k_range,score))
#now plotting the relationship between K and testing accuracy

import matplotlib.pyplot as plt
plt.plot(k_range,score)
plt.xlabel('Value of K')
plt.ylabel('Accuracy')
plt.show()

#training accuracy rises as model complexity increses
#testing accuracy penalizes model that r too complex or not complex enough
#for KNN models complexity is determined by the value of k
#lower K = more complex
#see K value rises from 6 and keeps highest till 16 and falls down

#Re testing with highest accuracy derived above

knn = KNeighborsClassifier(n_neighbors=11) 
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test) 
knn.predict([[3,5,4,2]])
X_new = [[3,5,4,2],[5,4,3,2]]
knn.predict(X_new)

#downsides of Train Test model is it provoides a high variance estimate of out of 
#sample accuracy meaning it can change a lot depending upon which set of observation is
#in the training set vs testing
#there is another evaluation model which is better. That is called Kfold

#video = 6
#https://www.youtube.com/watch?v=3ZWuPVWq7p4&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=6
#Data science in Python: pandas, seaborn, scikit-learn
#Model - linear regression
#openning CSV and setting a specific column as the index using index_col parameter
#see the un-named col is now set as the index
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Advertising.csv',index_col=0)
data.head()
data.shape

#predict sales based on asdvertisibg dollers

#becauae our response values are continuous it is a regression problem

#Using Sea born library to get a better feel of our data. This is built on top of matplotlib

# LINEAR REGREESSION
#PROS: fast, no tuning required, highly interpretable, well-understood
#CONS: unlikely to produce the best predictive accuracy (presumes a linear relationship btwn the features (dependant var) and response(independant var))
#in the real world often the relationship btwn feature and response is non linear
#where this model doesnt give as much predictive accuracy
import seaborn as sns
import matplotlib.pyplot as plt

#often the first relationship u want to visualize is the relation between each of the features and the response var
# this can be done by seaborn's pairplot function that produces pairs of scatterplots of each x and y pairs that u specify

sns.pairplot(data,x_vars=['TV','Radio','Newspaper'], y_vars = 'Sales',size=7, aspect=0.7)

#using seaborn to add a line of best fit as well as a 95% confidence band
sns.pairplot(data,x_vars=['TV','Radio','Newspaper'], y_vars = 'Sales',size=7, aspect=0.7,kind='reg')

#because there are traits from the model to show leanearity between each feature and response 
#this is a great
#candidate for linear regression

#Regression is simplfy a type of supervised machine learning where the response is continuous
#Linear regression is a particular ML model that can be used for regression problems that 
#happens to have the word 'regression' in it
#runs very fast
#no tuning required like classification
#easily undwerstand the model
#studied for many years and the model is easily understood

# Funbctional form of Linear Regression

#y = B0+B1XI+B2X2-----+BNXN
#B0 is the intercept
#BI,B2---BN - COEF
#x = feature/ independant var
#y=response/dependant var
#In this case:
#y=B0+B1*TV+B2*Radio+B3*nEWSPAPER

#SKIKITLEARN expects features and response (x and y) to be numpy arrays
#defining feature metrics

feature_cols=['TV','Radio','Newspaper']
#use this list to create a subset of the original dataframe
X=data[feature_cols]
# or same as below
X = data[['TV','Radio','Newspaper']]

# since pandas is built on top of numpy there is a numpy array storing the dataframe
# so our x can be pandas dataframe and our y can be pandas series
Y = data['Sales']

from sklearn import datasets,linear_model,metrics
reg=linear_model.LinearRegression()
from sklearn.model_selection import train_test_split
#SPLITTING INTO X AND Y TRAININ SETS
#defaukt split is 75%
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=1)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
reg.fit(X_train,Y_train)
# coef and intercept are attributes in linear regression which is why we can use with. and no
#parenthesis. _ is a skikit learn convention for any attributes that were estimated from the 
#data
reg.coef_
reg.intercept_

# Using Zip function to pair each of the feature name  from the features cols list
# with their coffecients

list(zip(feature_cols,reg.coef_))
#or
dict(zip(feature_cols,reg.coef_))
#output: [('TV', 0.04656456787415028),
# ('Radio', 0.17915812245088836),
# ('Newspaper', 0.0034504647111804347)]

# =============================================================================
# it means:
#     
#     y(sales) = 2.88(intercept sum of 3 features) + TV * 0.046 (coef for TV) + Radio * 0.179
#    (coef for Radio) + Newspaper * 0.003 (coef for Newspaper)
# 
# itemeans: example of TV - will be similar to Radio and newspaper:
#     for a given amount of radio and newspaper spending, a increases in 1 unit of TV increases:
#     spending of 0.046 unit of sales
#     or more clearly
#     for a given amount of radio and newspaper spending, a increase in 1000 dollars spent on
#     TV adds is associated with increase in sales of 46.6 items
#     note this is association/correlation not causation
#     if the coeff of TV was negative then it would interprate that an increase in 1000 dollars spent on
#     TV adds is associated with decrease in sales of 46.6 items   
# =============================================================================
    
    
    

    
# this level of interpretebility makes linera regression so helpful

#linear regression coeff can be also negative

y_pred = reg.predict(X_test)
y_pred

# evaluation metrics in Linear regression

#listen carefully from 25 min in the video

print('Root Mean squared error:',np.sqrt(metrics.mean_squared_error(Y_test,y_pred)))

#Feature selection

#train , test, split helps us to choose between the features . when we visualised the date
# newspaper we say had a very weak correlation. so lets try removing that from the model
feature_cols=['TV','Radio']
X=data[feature_cols]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=1)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
reg.fit(X_train,Y_train)
reg.coef_
reg.intercept_
list(zip(feature_cols,reg.coef_))
y_pred = reg.predict(X_test)
y_pred
print('Root Mean squared error:',np.sqrt(metrics.mean_squared_error(Y_test,y_pred)))

# see now the RMSE has reduced

#video 7
#selecting the best model in sklearn using cross validation


#what is the draw back of train/test split procedure for model evaluation
#how doesn k fold cross validation overcome the limitation
#how can cross validation be used to select tuning parameters, choose btwn models and
#select parameters
#whar r some possible improvements in cross validation
#using model_selection here instead of cross validation because i am getting error with cross 
#validation and can't update the terminal with 'conda update scikit-learn'
#or 'conda install -U scikit-learn'

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
iris = load_iris()
type(iris)
print(iris.data)
print(iris.feature_names)
print(iris.target)
print(iris.target_names)
X = iris.data
y = iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y, random_state=4)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred))

#The accuracy score is 0.97

X_train,X_test,y_train,y_test=train_test_split(X,y, random_state=2)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred))

#The accuracy score is 1.0

# =============================================================================
# this is why the train test split has so high variance
# what if we created a bunch of train test splits, calculated the testing accuracy for 
# each, and averaged the result together?
# Thats the essence of cross validation
# 
# steps for K fold cross validation
# 1) split the data set into k equal portions or folds/partitions
# 2) Use fold 1 as a testing set and the union of other folds as training sets
# 3) Calculate the testing accuracy
# 4) Repeat step 2 and 3 K times, using a different fold has the testing set each time
# 5) Use the average testing accuracy as the estimate of out of sample data    
# 
# see here the training set, the test set, and iterate the combination 
#Ktimes (Kfold) value both changes and the best combination is taken
#This K has no bearing with Knearest KNN hyperparameter: This Kfold - K is the cv 
#parameter in the score_val funtion

#Note: We are dividing the observations into Kfold iteraions but we are not dividing the
#features at this point
#see diagram on 7:12 min
# =============================================================================
#simulate splitting a data set of 25 observations into 5 folds of testing
#training data set and the iteration # gives the fold number

# =============================================================================
# below uses {} placeholder algning method of string where < means left justified
# > means right justified, :^ means center, also note that :^61 or :^9 or any nunbre
# means the space aligned 
# .format() passes the parameters into the placeholders {} { } etc..in the placehoolders
# you algn the spec by :<,:^, or:> and the number 
# =============================================================================
from sklearn.model_selection import KFold
import numpy as np
data = np.arange(25)
data
kf = KFold(n_splits=8, random_state=7)
#kf = KFold(data, n_splits=5, shuffle=False, random_state = 1)

# =============================================================================
# print the content of each training set and testing set
# data set contains 25 observations 0 to 24
# 5 fold cross validation meaning 5 iteration
# for each iteration - each obseration is either in training set or testing set but not both:
# every obseravtion is the testing set only once    
# print('{}{:^61}{}'.format('Iteration','Training set Observations','Testing set observations'))
# =============================================================================
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
#https://machinelearningmastery.com/k-fold-cross-validation/
for train, test in kf.split(data):
    print('train: %s, test: %s' % (data[train], data[test]))

kf.get_n_splits(data)    

#~~~~~~~~~~


kf = KFold(n_splits=5, shuffle=False, random_state = 1)
for train, test in kf.split(data):
    print('train: %s, test: %s' % (data[train], data[test]))

kf.get_n_splits(data)   

# =============================================================================
# advantages of cross validation
# more acurate estimate of out of sample accuracy
# more efficient use of use. every data is used for both testing and training
#all it gives is score which makes it difficult to understand using confusion matrix or ROC curve
# advantages of train-test spilt
# runs k times faster than cross validation
# simple to examine the detailed result in the testing process
# =============================================================================
    
 #Recommedation
# KFold or K=10 is generally generally based on past experiment
# for classification sampling its recommended that u use stratified sampling to create the folds
#each response class is represented with equal propostions in each of the kfolds
#scikit learns cross_val_score does all this by default
 
 
# example of using cross validation in parameter tuning of a classication problem
#iris_data set
 #goal is also select the best tuning parameter
from sklearn.datasets import load_iris
from sklearn import metrics
iris = load_iris()
X = iris.data
y = iris.target
from sklearn.model_selection import cross_val_score 
#10 fold cross validation with K=5 for KNN(the n-Neighbprs parameter)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
#note here we dont have to split the data
#the stratified folds splits it
scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy')
print(scores)
print(scores.mean())
#search for a optimal value of KNN
# see here for each KNN value from 1 to 31, the cross val runs 10 times each totalling 31 numpy arrays with 10 scores each

k_range=range(1,31)
k_scores=[]
for k in k_range:
    KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy')
    k_scores.append(scores)
print(k_scores) 
print(len(k_scores))   
    
#we are averaging every 10 which is why its producing a 31 length numpy array
from sklearn.datasets import load_iris
from sklearn import metrics
iris = load_iris()
X = iris.data
y = iris.target
from sklearn.model_selection import cross_val_score 
from sklearn.neighbors import KNeighborsClassifier
k_range=range(1,31)
k_range
k_scores=[]
for k in k_range:
    print('K:',k)
    knn=KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy')
    print(scores.mean())
    k_scores.append(scores.mean())
print(k_scores) 

# very diff to see from the numpy array so plotting to get the cross validation accuracy 
# for the optimal KNN
import matplotlib.pyplot as plt
plt.plot(k_range,k_scores)
plt.xlabel('KNN value')
plt.ylabel('mean cross validation score for each KNN')
# =============================================================================
# see the graph is an upside down U which is quite typical when examiming the 
#relationship between
# model complexity parameter and mpdel acuracy
# The highest values of KNN are 13 to 20
# This is an example of the bias-variance trade off in which low value of KNN
# produces a model with low bias and high variance and a high value of KNN produces a 
# model with high bias`and low variance
# The best model is found somehwre in the middle because it balances bias and variances
#and this its most likely to generalise the out of sample data
#when we chose the value KNN we would chose value that gives the lowest complexity model/less noise
#this we choose 20 for which the accuracy is also high and is kind of in the moddle

# =============================================================================

# example of using cross validation to choose the best type of model
# comparing the best KNN model that we found in iris data et tuning and we will compare it with
#logistic regression model

#knn with 20 and kfold =10

knn=KNeighborsClassifier(n_neighbors=20)
print(cross_val_score(knn,X,y,cv=10,scoring='accuracy').mean())

# gives a mean accuracy score of .98

# now with logreg

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
print(cross_val_score(logreg,X,y,cv=10,scoring='accuracy').mean())
# gives a mean accuracy score of .953

#thus we conclude that KNN with 20 is a better model to choose 


# now applying cross validation to select feature in linear regression
import numpy as np
import pandas as pd
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Advertising.csv',index_col=0)
data.head()
data.shape
feature_cols=['TV','Radio','Newspaper']
X=data[feature_cols]
# or same as below
#X = data[['TV','Radio','Newspaper']]
y = data['Sales']
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
reg=linear_model.LinearRegression()
scores = cross_val_score(reg,X,y,cv=10,scoring ='neg_mean_squared_error')
print(scores)

# this code doesnt runs here for reason unknown to me
# however ideally it should gove numpy with negative scores which is wrong
#bcoz mean_squared_error cannot be nagtive
#workaround of these negative values
mse_scores = -scores
print(mse_scores)
#calculate mean of all the 10 folds
print(mse.scores.mean())
#10 fold validation with 2 featurss exclusing newspare
#just change the X --->feature_col = ['TV','Radio'] taking out newspaper

#video 8
#Selecting the best model in scikit-learn using cross-validation
#how can kFOLD 

#this is a contnbuation to the tuning example of KNN using foreloop in the iris
#example
#More efficient parameter tuning using GridSearchCV -allows you to define a grid
#of parameters that will be searched using the K-fold cross validation
#GridseachCv automates the forloop that searches the best tuning parameter
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
iris = load_iris()
X = iris.data
y = iris.target
k_range=np.array(range(1,31))
k_range
#create a parameter grid: map the parameter names to the values that should be searched
#this is a python dict which has the single key - n_neighbors
param_grid=dict(n_neighbors=k_range)
print(param_grid)
knn=KNeighborsClassifier(param_grid)
#instantiate the grid: This replaces the for loop of iterating the KNN values
#this grid object can do 10 fold cross validation using the knn model and classification
#accuracy as the evaluation criteria. In addtion it has been given the param grid
#which is a dict with KNN values from 1 to 30
grid = GridSearchCV(knn,param_grid,cv=10,scoring='accuracy')
#fitting the model with data
grid.fit(X,y)
grid.cv_results_
#view the complete results(list of tuples)
#to print the results - convert the search results into a pandas DataFrame
results = pd.DataFrame(grid.cv_results_)
results
results.columns
# view the mean and standard deviation of the test scores for each set of parameters
results[['mean_test_score', 'std_test_score', 'params']]
# examine the results of the first set of parameters
results['params'][0]
#or
results.loc[0,'params']
results['mean_test_score'][0]
# list all of the mean test scores
results['mean_test_score']

# if the std_test_score which is the standard deviation of each of the meam score is 
#high it means that the cross validation accuracy is low

#create a list of mean sores only
#plot the results

import matplotlib.pyplot as plt
plt.xlabel('KNN value')
plt.ylabel('mean cross validation score for each KNN')
plt.show()

#examine the best model
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)


#searching multiple parameters simultaneously

# in case of KNN another parameter that might be worth tuning other than K (knn) 
#is the weights parameter which controls how the Knearest neighbors are weighted when making a 
#prediction. The default option is uniform which means that all points in the neighborhood
#are weighted equally but another option is distance which weighs closer neighbors more
#heavily than farther neighbors
#Therefore we are going to create another option called weight option in addition'
#to the KNN

k_range = np.array(range(1,31))
weight_option = ['uniform','distance']
param_grid = dict(n_neighbors = k_range,weights = weight_option)
print(param_grid)
knn=KNeighborsClassifier(param_grid)
grid = GridSearchCV(knn,param_grid,cv=10,scoring='accuracy')
grid.fit(X,y)
results = pd.DataFrame(grid.cv_results_)
results[['mean_test_score', 'std_test_score', 'params']]
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
# see with the weight option nothing improved much
#KNN remains 13, the weight parameter is the Uniform - which is the default

#WHAT TO DO WITH THE BEST PARAMETER TO MAKE PREDICTION
#before you predict using the model you train the model with the best known parameters
# that you just found with gridsearchCV. otherwise u will be throwing away potentially thowing
#best fitted model
#therefore gridsearch cv is most useful when u run it before making predictions

knn=KNeighborsClassifier(n_neighbors=13,weights='uniform')
knn.fit(X,y)
#make a prediction on out of sample data
knn.predict([[3,5,4,2]])

# or using GridSearchCV which automaticlly fits the best model

grid.predict([[3,5,4,2]])

#reducing computation expense of grid search cv by using RamdomizedSearchCV
# =============================================================================
# the problem that randomized searchCV aims to solve is:
#     when you are performing an exhaustive search using many different parameters at once
#     it might quickly become computationally infeasable
#     for searching 10 different parameters for 10 diff values 10 fold will easily become 10000
#     searches.
# Ramdomzed CV solves this proble by reduces this computational crisis by searching a ramdomly selected subset of
# different parameters. Thus u control the computational budget    
#it also allows to explicitly control the number of parameter combinations that are attempted
# =============================================================================
from sklearn.model_selection import RandomizedSearchCV
weight_option = ['uniform','distance']
#in ramdomizedsearchcv u specifiy the parameter dictribution rather than param_grid
#here param_grid and param_dist are similar because we are specifying dicete values list
#integers and string
#however if one of your tuning parameters in continous such a regularization parameter for
# a regression problem , it is important to soecify a continious distribution rather than
# a list of possible values so randomized search cv can perfom a more finegrained search
param_dist = dict(n_neighbors = k_range,weights = weight_option)
print(param_dist)
knn=KNeighborsClassifier(param_dist)
#n_iter controls the number of random searches
rand = RandomizedSearchCV(knn,param_dist,cv=10,scoring='accuracy',n_iter=10,random_state=5,return_train_score=False)
rand.fit(X,y)
results = pd.DataFrame(rand.cv_results_)
results.columns
results[['mean_test_score', 'std_test_score', 'params']]
#see even though it searched just 10 combinations it manaded to find a best score
#like gridseacrhcv
print(rand.best_score_)
print(rand.best_params_)
print(rand.best_estimator_)
#experimet to show how randomsiex seatch cv manages to get the max score almost always

best_score=[]
for i in range(20):
    rand = RandomizedSearchCV(knn,param_dist,cv=10,scoring='accuracy',n_iter=10,random_state=5,return_train_score=False)
    rand.fit(X,y)
    best_score.append(round(rand.best_score_,3))
best_score    
    
# contininous distribution in param_dist 
#https://github.com/amueller/pydata-nyc-advanced-sklearn/blob/master/Chapter%203%20-%20Randomized%20Hyper%20Parameter%20Search.ipynb


#video 9
# =============================================================================
# 
# Agenda
# what is the purpose of model evaluation and what are some of the common procedures
# what is the usage of classification accuracy and what are some limilations
# how does confusion matrix describe the perfomace of a classifier
# what matrix can be computed from confusion matrix
# how can u adjust the classifier performace by changing the classification threshold
# what is the purpsoe of ROC curve
# how does AUC(Area) under the curve differ from classification accuracy
# =============================================================================

# =============================================================================
# review of model evaluation
# need a way to choose between models: differemt model types, tuning parameters, and features
# use a model evaluation procedure to generalise how well a model can generalize btween 
#out of sample data
# require a model evaluation metric to pair up with your model evaluation procedure and 
#quantiy model performace
# 
# Model evaliation procedures
# 
# 1) Train/test on same data
#     rewards overly complex models that over fit the data and doesnt generlize well on out of
#     same data even though the accuracy score is high
# 2) Train/Test split
#    split dataset into 2 pieces - one used for training, other for testing
#    better estimate of out of sample data but has high variance estimate
#    useful due to speed, simplicity and fexibility
# 
# 3)K-fold cross validation
#   systemically creates K train test splits and average the results together
#   better estimte of out of sample
#   runs K times slower than train test split
#   
# Model evaluation metric
#  Regression problems: MEAN SQUARED ERROR, ROOT MEAN SQUARED ERROR, MEAN ABSOLUTE ERROR
#  classification problem: classification ACCURACY 

#Other classification metrics are focus of this video
#     
# =============================================================================

import pandas as pd
url='https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
col_names = ['pregnant', 'glucose','bp','skin','insulin','bmi','pedigree','age','label']
pima = pd.read_csv(url,header = None,names =col_names)
pima.head()

#can we predict the diabetes status of a patient given the health measurements
#selecting the feature columsn

feature_cols = ['pregnant','insulin','bmi','age']
X=pima[feature_cols]
y=pima.label

#split X and Y in train-test data sets

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

#train a logictoc regression model on the training dataset

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)

#make class predictions for the testing set

y_pred_class = logreg.predict(X_test)

#classification accuracy: percentagge of correct predictions
df = pd.DataFrame({'Actual data':y_test,'Predicted Data':y_pred_class})
df  
logreg.score(X_test,y_test)
from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred))
len(y_test)
type(y_test)
import numpy as np
y_test1=np.array(y_test)
type(y_test1)
len(y_pred_class)
type(y_pred_class)
# or

from sklearn import metrics
print(metrics.accuracy_score(y_test1,y_pred_class))


# any time u measure a classification accuracy its always good that u also measure 
#the null accuracy
#which means how much accuracy could have been achieved by always predicting 
#the most frequent response 
#class correctly
#Null accuracy: accuracy that could be acheived by always predicting the most frequest class

#examine the response class distribition - which the distributuon of 0 and 1 of the testing set
#using a pandas series method
y_test.value_counts()
#calculate the percentage of 1s
#here the mean method worked because 0....+1....../total# of 0 and 1
#62/192 = .32
#calculte the percentage of ones
y_test.mean()
#calculte the percentage of zeroes - thi is the null accuracy
print(1 - y_test.mean())

#when we compare the model accuracy of 69% with the null accuracy of 67% - our
#model doesnt look very good

#this is a big weakness of the classificztion accuracy metrics because it doesnt tells us
#anything about the underlying distribution of the testing set and therefore the true 
#output and the predicted output

#calculate the null accuracy of binary classification problems coded as 0/1 in a single line of code
#this will only work for binary classifixation problem 
max(y_test.mean(),1 - y_test.mean())
# .677 is the null accuracy for this problem

# the following will work for multi class classification(for a problem with 3 or more classes)
#problem though it will only work if y_test is a pandas series
# head (1) as it gives the number of zeroes
y_test.value_counts().head(1)/len(y_test)
#in other words a dumb model which always predicts that the patient doesnt have diabetes
#is corerect 68% of the time
#this is not a useful model but it can serve as a baseline agianst which we might want to
#measure our logistic regression model
#when we measure the null accuracy of 68% with a model accuracy of 69% - our model doesnt look
#very good
# =============================================================================
# this demonstrates one weakness of the classification accuracy as a model evaluation metrics
# in that the classification accuracy doesnt tell u anything about the underlying distrution of the 
# testing set
# =============================================================================

# =============================================================================
# one other weakness of classification accuracy
# lets look at the first 25 response of the response values
# see the models always correctly predicts a 0 but rarely predicts a one
#meaning the model is predicting certain types of errors more than others
#it doesnt tell u about the underlying distrubution of response values which we have to
#calculate by null accuracy
#it doesnt tell u about the types of error the model is making which is use ful to know
#this issue is best addressed by the confusion metrics
# =============================================================================
print('True:',y_test[0:25])
print('Pred:',y_pred_class[0:25])
df1 = pd.DataFrame({'True':y_test[0:25],'Pred':y_pred_class[0:25]})
df1
# =============================================================================
#        Confusion matrix
#        
# A table that describes the performace of a classification model
# every observation in a testing set is represented in exactly a box
# its a 2x2 matrix because there are 2 response classes
# CONVENTION - to describe that the class encoded as 1 as positive class
#and 0 as negative class
# Basic terminology
# ACTUAL-PRED --- 1-1/0-0/1-0/0-1
# True Positive (TP): We CORRECTLY predicted that they DO have diabetes 1-1 (15 cases)
# True Negative (TN): We CORRECTLY predictly that they DONT have diabetes  0-0 (118 cases)
# False Positves (FP):  We INCORRECTLY predicted that they DO have diabetes   (aka: Type 1 Error) (0-1 -12 cases)
# False Negatives (FN):  We INCORRECTLY predicted that they DONT have diabetes (aka: Type 2 Error) (1_0 -47 cases) 
# 
# 
# =============================================================================

#IMP: first agrumet is true valus,second argument is predicted values
#x`output a 2x2 numpy array which is not labelled with text (refer to confusion_matrix diagram in python folder)
# =============================================================================
# based on the diagram: probability that the model will have TN is 90% 
#                       probability that the model will have TP is 24% 
# 
# 
# =============================================================================
# If there were 5 possible response classes it wud be a 5x5 MATRIX
# =============================================================================
#finally its iportant to know these confision mnatrix are numbers of integre counts not rate
# below are some of the rates that can be calculated using confusion matrix

#=============================================================================
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


def print_estimator_name(estimator):
    print(estimator.__class__.__name__)
print_estimator_name(LogisticRegression())
    
logreg.fit(X_train,y_train)
y_pred_class = logreg.predict(X_test)
from sklearn.metrics import confusion_matrix
print(metrics.confusion_matrix(y_test,y_pred_class))

# =============================================================================
# save confusion metrics in an object called confusion and slice each confusion
# in a separate numpy array
#refer to diagram: detailed_confusion
# =============================================================================

# =============================================================================
# 
# The confusion metrics helps you to understand the performace of your classifier
# but it doesnt help u to choose the model It is mot a model evaluation metric so u cant simply
# tell sklearn to chose the model with the best confusion metrics
# =============================================================================

# =============================================================================
# however there are many metrics that can be calculated from confusion metrics and those 
# can be used to chose the model
# =============================================================================
confusion = metrics.confusion_matrix(y_test,y_pred_class)
TP = confusion[1,1]
TN = confusion[0,0]
FP = confusion[0,1]
FN = confusion[1,0]

# =============================================================================
# as u now know classification accuracy can be calculated as: TP+TN/float(TP+TN+FP+FN)
# use float in deno to compute true division
# 
# =============================================================================
print((TP+TN)/float(TP+TN+FP+FN))

# IS EQUAL TO

print(metrics.accuracy_score(y_test,y_pred_class))

# =============================================================================
#     Classification error aka: Misclassification Rate
#  
# how often the clarrifier is incorrect    
# see below
# =============================================================================

print((FP+FN)/float(TP+TN+FP+FN))

# IS EQUAL TO

print(1 - metrics.accuracy_score(y_test,y_pred_class))


# =============================================================================
#               Sentivity
# when the actual value is posiive - how often is the prediction correct - TP
# how sensitive is the classifier in detecting those positive instances
# Also known as 'true positve rate' or 'Recall'

#see below               
# =============================================================================

print(TP/float(TP+FN))

# IS EQUAL TO

print(metrics.recall_score(y_test,y_pred_class))

# =============================================================================
#               Specificty
# when the actual value is negaiive - how often is the prediction correct - TN
# how specific is the classifier in detecting those negative instances


#see below               
# =============================================================================

print(TN/float(TN+FP))

# False positive rate: when the actual value is negaiive - how often is the 
#prediction incorrect - FN
#False posiive rate is 1 - specificity

print(FP/float(TN+FP))

#Precision: what percentage of predictions are correct considering the total 
#number of positive 
#predictions made
print(TP/float(FP+TP))

# ie equal to

print(metrics.precision_score(y_test,y_pred_class))

# =============================================================================
# Many other metrics can be calculated from confusion metrics: F1 Score, Mathews correlation
# coeff etc.
# 
# Confusion:
#     confusion matrix gives u a more complete picture of how a classifier is performing
#     also allows u to compute various classification metrics and these metrics can guide
#     your model selection
# which metrics should u use:
# choice of metrics depend on ur business objective
# spam filter (positve class is spam): optimize for precision or specificity because 
# false negatives (spam (1)predicted as Non -spam (0) and goes to ur inbox) is more acceptable than
# false positives (Non spam is predicted as spam and caught in the spam filter) 
# fraud transaction detector (posotve class is fraud) optimize for sensitivity because false 
# postives (normal trans flagged as frauds) are more acceptable than false negatives (fraud trans that
#          are not detected)
# =============================================================================

# =============================================================================
#    ADJUSTING THE TRANSACTION THRESHOLD
# =============================================================================
                     
#print the first 10 predicted response

logreg.predict(X_test)[0:10]

#print the first 10 predicted probabilities of class membership
# =============================================================================
# for each row this number add up to 1:
#     the left row is the preidcted probablity that each observation is a member of class0
#     the left row is the preidcted probablity that each observation is a member of class1
# where do this numbers come from?
# this can be explained by how logistic regression works but at highlevel
# the model learns the coeff from each input feature from the model training process and those
# coeffs are used to calculate the likely of each class for each observation in the test set
# 
# since this model predicts the likelihood of diabetes using the above - we might rank observations
# by predicted probability of diabetes and prioritise our patient preventative outreach accordingly
# since it makes more sense to contact and give case to a 95% diabetes than 55%diabetes 
# 
# when u run the predict method of classification model it first predicts the probablities of each
# class and then chooses the class with the highest probablity as the predicted probablity :
#     
# for a binary problem like this one another way of thinking abt it there is a 0.5 threshold
# and class 1 is predicted only when 0.5 threshold is crossed, otther wise 0 is predicted
# 
# 
# =============================================================================
logreg.predict_proba(X_test)[0:10,:]   

# =============================================================================
# 
# lets now isolate the predicted probablity of class 1 because knowing that alone 
# enables u to calculate the predicted prob of both classes -
# predict first probalities of 1 (also column 1)    
# 
# =============================================================================

logreg.predict_proba(X_test)[0:10,1]       
                  
#storing the predicted prob of 1s

y_pred_prob = logreg.predict_proba(X_test)[:,1]
type(y_pred_prob)
y_pred_prob.shape

#plot a histogram of these probabilities to help demonstrate how adjusting
# =============================================================================
# the classification threshold can impact the performace the model
# first we need to allow plots to appear and we are overriding one of the default
# matplotlib settings
# will use matplotlib to plot a histgram of the predicted probabilities of class 1
# 
# =============================================================================
import matplotlib.pyplot as plt
#plt.rcParams['font.size']=14
plt.hist(y_pred_prob,bins=8)
plt.xlim(0,1)
plt.title('histogram of predicted probability')
plt.xlabel('predicted probability of diabetes') 
plt.ylabel('frequency')    
# =============================================================================
# 
# seeing the graph above u see that almost 45 people had probablity btn 0.3 to 0.35
# which our model didnt see has 1. If we were to change the threshold then we can adjust both sensitivity and specificity of 
# the model by adjusting the threshold    
# =============================================================================
#decrease the threshold for predicting diabetes to increase sensitivity of the 
#classifier 
# =============================================================================
# now we new theshold of 5 all probabilities above 0.3 are now predicted as class 
#1 which increases
# sensitivity of the model because the classifier is now more sensitive to positive 
#instances
# for doing that:
#     we can use binarize function from sklearn preprocessing that will return 1 
#for all values
#     above 0.3 and a 0 otherwise
#     the results are in a 2 dimensional numpy array and we slice out the first 
#dimension using the
#     bracket notation and save the results in the y_pred_class object
# =============================================================================
y_pred_prob

import numpy as np    
from sklearn.preprocessing import binarize
# has to convert it to a 2D array because binarize function expects it to be
#done by reshaping
y_pred_class = binarize(y_pred_prob.reshape(1,-1),0.3)
y_pred_prob[0:10] 
y_pred_prob[0:10] 
#now converting it back to a 1D array by ravel because confusion matrix wants to be 1D array
y_pred_class1 = y_pred_class.ravel()
y_pred_class1

type(y_pred_class1)

#below code can also convert the 1D panda series to a 1d array
#type(y_test)
#y_Test = np.array(y_test)
#y_Test
#type(y_Test)
#print new confision matrix with 0.3 threshold
print(metrics.confusion_matrix(y_test,y_pred_class1))
print()


#sensitivity has increseesd (increased from 24 to 74)

print (46/float(46+16))

#specificity has decresed(from 91% to 61%)

print(80/float(80+50))

#concluded - sensiticty and specificity has a inverse relationshop
#remember drcreasing the threshold will be one of the last steps u should take in
#model building process
#The majority of the time should be on building better model and seleting the best possible model


# =============================================================================
#    ROC Curves and area under the curve (AUC)
#    
#    wouldn't it be nice if we can check how sensitivity and specificuty are affected by
#    various thresholds without changing the threshold?
#    Answer is to plot ROC and AUD
#    ROC curve can help you to choose a threshold that balances sensitivity and specificity
#    in a way that makes sense for ur particular context
#    though u cant atually see the threshold used to genrate the curve on the ROC curve iself
# =============================================================================


# first arg is true values, second arg is predicted
#false positive rate
#true positive rate
#dont take y_pred_class or class1 bcpz it will generate incorrect result with giving error

fpr,tpr, thresholds = metrics.roc_curve(y_test,y_pred_prob)
plt.plot(fpr,tpr)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False positive rate(1-specificity)')
plt.ylabel('True positive rate(sentisivity)')
# a ROC curve helps u visually choose a threshold that balances sensitivity and 
#specificity

# a small function written below to pass a value of the threshold to evaluate 
#sensitivity and specificyt

def evaluate_threshold(threshold):
    print('sensitivity:',tpr[thresholds>threshold[-1]])
    print('specificity:',1-fpr[thresholds>threshold[-1]])
    
evaluate_threshold(1/2)


# Area under the ROC curve
#because an ideal classifier wud hug the upper left corner of the curve, a bigger area of AUC
# =============================================================================
# #is indicative of a better over all classifier
# #AUC is a single number unit that summarizes of the performance of a classifier as an alternative to 
# classification accuracy
# =============================================================================

#first arg is true val, second is pred

print(metrics.roc_auc_score(y_test,y_pred_prob))

#if you randomly chose one postive and one negative observation, AUC represents 
#the likelihood that the classifier will assign a higher predicted probability to the
# =============================================================================
# postive observation
# AUC is useful when there is a highclass imbalance (unlike classicifcation accuracy)
# =============================================================================
#calculate cross validated AUC

from sklearn.model_selection import cross_val_score
cross_val_score(logreg,X,y,cv=10,scoring='roc_auc').mean()

#-----Decision Tree -----from Kevin but in GitHub
#https://github.com/justmarkham/DAT8/blob/master/notebooks/17_decision_trees.ipynb

import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/vehicles_train.csv'
train = pd.read_csv(url)
train
train['prediction'] = train.price.mean()
train

# calculate RMSE for those predictions
from sklearn import metrics
import numpy as np
np.sqrt(metrics.mean_squared_error(train.price, train.prediction))

# define a function that calculates the RMSE for a given split of miles
def mileage_split(miles):
    lower_mileage_price = train[train.miles < miles].price.mean()
    lower_mileage_price
    higher_mileage_price = train[train.miles >= miles].price.mean()
    higher_mileage_price
    train['prediction'] = np.where(train.miles <miles , lower_mileage_price, higher_mileage_price)
    train['prediction']
    return np.sqrt(metrics.mean_squared_error(train.price, train.prediction))

# calculate RMSE for tree which splits on miles < 50000
print ('RMSE:', mileage_split(50000))
train

# calculate RMSE for tree which splits on miles < 100000
print ('RMSE:', mileage_split(100000))
train
## check all possible mileage splits

mileage_range = range(train.miles.min(), train.miles.max(), 1000)
RMSE = [mileage_split(miles) for miles in mileage_range]
RMSE
import matplotlib.pyplot as plt
plt.plot(mileage_range, RMSE)
plt.xlabel('Mileage cutpoint')
plt.ylabel('RMSE (lower is better)')

#Recap: Before every split, this process is repeated for every feature, and 
#the feature and cutpoint that produces the lowest MSE is chosen.
import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/vehicles_train.csv'
train = pd.read_csv(url)
train
train['vtype']
train['vtype'] = train.vtype.map({'car':0, 'truck':1})
train['vtype']
feature_cols = ['year', 'miles', 'doors', 'vtype']
feature_cols
X = train[feature_cols]
X
y = train.price

## instantiate a DecisionTreeRegressor (with random_state=1)
from sklearn.tree import DecisionTreeRegressor
treereg = DecisionTreeRegressor(random_state=1)
treereg
## use leave-one-out cross-validation (LOOCV) to estimate the RMSE for this model

from sklearn.model_selection import cross_val_score
scores = cross_val_score(treereg, X, y, cv=14, scoring ='neg_mean_squared_error')
scores
np.mean(np.sqrt(-scores))

#What happens when we grow a tree too deep?
#The training error continues to go down as the tree size increases (due to overfitting), 
#but the lowest cross-validation error occurs for a tree with 3 leaves.

#Tuning the regression Tree

#Let's try to reduce the RMSE by tuning the max_depth parameter:
# try different values one-by-one
treereg = DecisionTreeRegressor(max_depth=1, random_state=1)
scores = cross_val_score(treereg, X, y, cv=14, scoring='neg_mean_squared_error')
np.mean(np.sqrt(-scores))

#Or, we could write a loop to try a range of values:

# list of values to try
max_depth_range = range(1, 8)

# list to store the average RMSE for each value of max_depth
RMSE_scores = []

# use LOOCV with each value of max_depth
for depth in max_depth_range:
    treereg = DecisionTreeRegressor(max_depth=depth, random_state=1)
    MSE_scores = cross_val_score(treereg, X, y, cv=14, scoring='neg_mean_squared_error')
    RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))
    
RMSE_scores    

# plot max_depth (x-axis) versus RMSE (y-axis)
plt.plot(max_depth_range, RMSE_scores)
plt.xlabel('max_depth')
plt.ylabel('RMSE (lower is better)')

treereg = DecisionTreeRegressor(max_depth=3, random_state=1)
treereg.fit(X, y)
## "Gini importance" of each feature: the (normalized) total reduction of error brought by that feature
pd.DataFrame({'feature':feature_cols, 'importance':treereg.feature_importances_})

# create a Graphviz file
from sklearn.tree import export_graphviz
export_graphviz(treereg, out_file='tree_vehicles.dot', feature_names=feature_cols)

#Making predictions with the testing data

# read the testing data
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/vehicles_test.csv'
test = pd.read_csv(url)
test['vtype'] = test.vtype.map({'car':0, 'truck':1})
test
#Question: Using the tree diagram above, what predictions will the model make for each observation?
feature_cols = ['year', 'miles', 'doors', 'vtype']
feature_cols
X_test = test[feature_cols]
y_test = test.price
y_pred = treereg.predict(X_test)
y_pred
#calculate RMSE
np.sqrt(metrics.mean_squared_error(y_test, y_pred))

# calculate RMSE for your own tree!
y_test = [3000, 6000, 12000]
y_pred = [0, 0, 0]
from sklearn import metrics
np.sqrt(metrics.mean_squared_error(y_test, y_pred))


#Buildiong a classification Tree 
import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/titanic.csv'
titanic = pd.read_csv(url)
titanic.head()
for i in titanic['Embarked']:
    if pd.isnull(i):
        print('happy')
titanic['Embarked'].value_counts().idxmax()
titanic.isnull()['Age']
set(titanic.Sex.unique()) 
encoding_sex = {

    'male': 1,

    'female': 0

}
encoding_sex.keys()
titanic['Sex'] = titanic.apply(lambda x: encoding_sex.get(x['Sex']), axis=1)
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
titanic.Embarked.fillna(titanic.Embarked.mode(),inplace=True)
titanic.isnull().sum()

#create dummy variable for Embarked

Titanic = pd.get_dummies(titanic,columns = ['Embarked'],drop_first=True)
feature_cols = ['Pclass', 'Sex', 'Age', 'Embarked_Q', 'Embarked_S']
X = Titanic[feature_cols]
y = Titanic['Survived']

# fit a classification tree with max_depth=3 on all data
import numpy as np
from sklearn.tree import DecisionTreeClassifier
treeClf = DecisionTreeClassifier(max_depth = 3, random_state = 1)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(treeClf, X, y, cv=14, scoring='neg_mean_squared_error')
scores
np.mean(np.sqrt(-scores))
treeClf.fit(X,y)
X['Age'].median(skipna=True)

# compute the feature importances
pd.DataFrame({'feature':feature_cols, 'importance':treeClf.feature_importances_})


# Unsupervised learning - where the response is not defined or categorized
# K Means Clustering, clustering evaluation, DBSCAN clustering

#https://github.com/justmarkham/DAT8/blob/master/notebooks/19_clustering.ipynb


import pandas as pd
beer = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/beer.txt', sep = ' ')
type(beer)
beer
# The name column wll not be useful
X = beer.drop('name',axis=1)
X
# trying the first clustering model

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
km.fit(X)

# finding the clusters

km.labels_

# or it can be found by

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
ypred = km.fit_predict(X)
ypred

beer['cluster'] = km.labels_
beer

# sorting the dataframe by cluster
beer.sort_values('cluster')

#there are 4 features and 3 clusters (we chose n_cluster = 3). So there will be 12 centers
#centers are basically these center points of each cluster-feature subgroups

km.cluster_centers_

# this is same as below
# calculate the mean of each feature per cluster
beer.groupby('cluster').mean()

## save the DataFrame of cluster centers
center = beer.groupby('cluster').mean()
center

# plotting the each features and their centers

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

import numpy as np
colors = np.array(['red', 'green', 'blue', 'yellow'])
colors

c = colors[beer.cluster]
c

# scatter plot of calories versus alcohol, colored by cluster (0=red, 1=green, 2=blue)

plt.scatter(beer.calories, beer.alcohol, c=colors[beer.cluster])
beer[['calories','alcohol','cluster']]
# using seaborn

import seaborn as sns
beer['cluster'] = km.labels_
sns.set_style('whitegrid')
sns.lmplot('calories', 'alcohol',data = beer[['calories','alcohol','cluster']], hue='cluster',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)


#cluster mean of each centers, marked by "+"
plt.scatter(center.calories,center.alcohol,marker = '+', color = 'black', s=50)
plt.xlabel('calories')
plt.ylabel('alcohol')

#scatter plot matrix (0=red, 1=green, 2=blue)

pd.scatter_matrix(X, c=colors[beer.cluster], figsize=(10,10), s=100)

#Repeat with scaled data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
km = KMeans(n_clusters=3, random_state=1)
km.fit(X_scaled)
beer['cluster'] = km.labels_
beer.sort_values('cluster')

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

#Do you notice any cluster assignments that seem a bit odd? How might we explain those?

# =============================================================================
# Part 2: Clustering evaluation
# The Silhouette Coefficient is a common metric for evaluating clustering 
# "performance" in situations when the "true" cluster assignments are not known.
# 
# A Silhouette Coefficient is calculated for each observation:
# 
# SC = {b-a}/ {max(a, b)}
# a = mean distance to all other points in its cluster
# b = mean distance to all other points in the next nearest cluster
# It ranges from -1 (worst) to 1 (best). A global score is calculated by 
# taking the mean score for all observations.
# =============================================================================

# calculate SC for K=3
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

# seems that n_cluster = 4 will have rge hightest score

# K-means with 4 clusters on scaled data
km = KMeans(n_clusters=4, random_state=1)
km.fit(X_scaled)
beer['cluster'] = km.labels_
beer.sort_values('cluster')

km.cluster_centers_  
km.inertia_