#importing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn
import pickle

df = pd.read_csv('hiring.csv')
# df = pd.read_csv('C:\\Users\\USER\\Desktop\\Hiring\\heroku_Flask_sal_deployment\\hiring.csv')

# df.isnull().sum()

#fill test_score null value with mean value
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].mean())

#Replace experience null values with 0
df['experience'] = df['experience'].fillna(0)

#function to convert experience string values to numbers
def strTonumbers(word):
  #dictionary
  dict = {0:0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11}
  return dict[word]

#String values of experience is converted to numbers
df['experience'] = df['experience'].apply(lambda x: strTonumbers(x))

# Divide data into x and y
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# # Spliting data into train and test data
# from sklearn.model_selection import train_test_split
# xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=5) 
#Since we have a very small dataset, we will train our model with all availabe data.

#Creating model and train dataset
from sklearn.linear_model import LinearRegression
lrmodel = LinearRegression()
# lrmodel.fit(xtrain,ytrain)
lrmodel.fit(x,y)

#Prediction
# ytest_pred = lrmodel.predict(xtest)
# ytest_pred

# Saving model to disk
pickle.dump(lrmodel, open('LRmodel.pkl','wb'))
# pickle.dump(model_name, open('save as filename','write mode'))


# Loading model to compare the results
lrmodel = pickle.load(open('LRmodel.pkl','rb'))
# model_name = pickle.load(open('save as filename','read mode'))
print(lrmodel.predict([[2, 9, 6]]))
# [53290.89255945]
# print(df)


# Path 
# cd C:\Users\USER\Desktop\Hiring\heroku_Flask_sal_deployment
#To run 
# python .\LRmodel.py