# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 19:37:41 2022

@author: Dell
"""
#importing important libraries
import numpy as py 
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,confusion_matrix

#read csv file
sal=pd.read_csv('income.csv',na_values=[" ?"])
#make a copy of a dataframe
sal_copy=sal.copy(deep=True)
#to find no. of null values in data
sal_copy.isnull().sum()
#sto see first summary of data
sal_copy.describe()
##sal_copy.describe(include=0)
sal1=sal_copy.dropna(axis=0)
sal.isnull().sum()
#Subsetting the rows to consider at least one column is missing
missing=sal[sal.isnull().any(axis=1)]
#Dropping NA values
sal2=sal.dropna(axis=0)
#to find relationship between independent variables
correlation=sal2.corr()
#to find column names
sal2.columns
#Step 1. Reindexing salary 0 and 1(0 for =<50000 and 1 for >50000)
#Step 2. mapping data in form of 0 and 1 using dictionary
sal2['SalStat']=sal2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
#remixing the salary status
newdata=pd.get_dummies(sal2,drop_first=True)
#storing column name
column_list=list(newdata.columns)
print(column_list)
#saperating input names from data
features=list(set(column_list)-set(['SalStat']))
#storing the output vales in y
y=newdata['SalStat'].values
print(y)
#storing the values from input features
x=newdata[features].values
print(x)
#splitting the data into train and test data
trainx,testx,trainy,testy=train_test_split(x,y,test_size=0.3,random_state=0)
#make a instance of model
logistic=LogisticRegression()
#fitting the values for x and y
logistic.fit(trainx,trainy)
logistic.coef_
logistic.intercept_
#prediction from test data
prediction=logistic.predict(testx)
print(prediction)
#drawing a confusion matrics 
confusion_matrix = confusion_matrix(testy,prediction)
print(confusion_matrix)
#calculating accuracy
accuracy_score = accuracy_score(testy,prediction)
print(accuracy_score)
#pinting the misclaassified values form prediction
print('misclassified_sample : %d' %(testy!=prediction).sum())

###regression after droping misclassified values
sal_copy['SalStat']=sal_copy['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(sal_copy['SalStat'])
cols=['gender','nativecountry','race','JobType']
new_data=sal_copy.drop(cols,axis=1)
new_data=pd.get_dummies(new_data,drop_first=True)
column_list=list(new_data.columns)
print(column_list)
features=list(set(column_list)-set(['SalStat']))
y=new_data['SalStat'].values
print(y)
x=new_data[features].values
print(x)
trainx,testx,trainy,testy=train_test_split(x,y,test_size=0.3,random_state=0)
logistic=LogisticRegression()
logistic.fit(trainx,trainy)
logistic.coef_
logistic.intercept_
prediction=logistic.predict(testx)
print(prediction)
confusion_matrix = confusion_matrix(testy,prediction)
print(confusion_matrix)
accuracy_score = accuracy_score(testy,prediction)
print(accuracy_score)
print('misclassified_sample : %d' %(testy!=prediction).sum())
#KNN Model
#Importing important libraries of KNN
from sklearn.neighbours import KNeighboursClassifier
#importing libraries for plotting
import matplotlib.pyplot as plt
#stprinf the k-nearest classifire
KNN_Classifier=KNeighboursClassifier(n_neighbours=5)
#fittingth value of x and y
KNN_Classifier.fit(trainx,trainy)
#predicting test values with model
prediction = KNN_Classifier.predict(testx)

#performance matrics check
confusion_matrix=confusion_matrix(testx, prediction)
print(confusion_matrix)
#Calculating accuracy
accuracy_score=accuracy_score(testy,prediction)
print(accuracy_score)
print('Misclassified samples: %d' %(testy!=prediction).sum())
#effect of k values on classifiers
Misclassified_Sample=[]
#calculating error for k values between 1 and 20
for i in range (1,20):
    Knn=KNeighboursClassifier(n_neighbours=i)
    Knn.fit(trainx, trainy)
    pred_i=Knn.predict(testx)
    Misclassified_Sample.append((testy!=pred_i).sum())
    print (Misclassified_Sample)
