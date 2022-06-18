#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 14:07:20 2022

@author: rajkumar
"""

# Import necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#% matplotlib inline

# The above command sets the backend of matplotlib to the 'inline' backend. 
# It means the output of plotting commands is displayed inline.
df= pd.read_csv("/Users/rajkumar/Desktop/data2.csv") 

# Exploratory data analysis
# View the dimensions of df
print(df.shape)

# View the top 5 rows of df
print(df.head())

# Rename columns of df dataframe
df.columns = ['A', 'B']

# View the top 5 rows of df with column names renamed
print(df.head())

# View dataframe summary
print(df.info())

# View descriptive statistics
print(df.describe())

# Declare feature variable and target variable
X = df['A'].values
y = df['B'].values

# Plot scatter plot between X and y
plt.scatter(X, y, color = 'blue', label='Scatter Plot')
plt.title('Relationship A and B')
plt.xlabel('A')
plt.ylabel('B')
plt.legend(loc=4)
plt.show()

# Print the dimensions of X and y
print(X.shape)
print(y.shape)

# Reshape X and y
X = X.reshape(-1,1)
y = y.reshape(-1,1)

# Print the dimensions of X and y after reshaping
print(X.shape)
print(y.shape)

# Split X and y into training and test data sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Print the dimensions of X_train,X_test,y_train,y_test
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#FInding Outliers
plt.figure(figsize=(5,5))
sns.boxplot(y='B',data=df)

#Multivariate method: Just I am taking data2.csv as a sample for my analysis
plt.figure(figsize=(8,5))
sns.boxplot(x='A',y='B',data=df)

# Fit the linear model
# Instantiate the linear regression object lm
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

# Train the model using training data sets
lm.fit(X_train,y_train)

# Predict on the test data
y_pred=lm.predict(X_test)

# Compute model slope and intercept
a = lm.coef_
b = lm.intercept_,
print("Estimated model slope, a:" , a)
print("Estimated model intercept, b:" , b)  

# y = 0.404762902 * x - 28.79482147
# That is the linear model.
# Predicting Advertising values
lm.predict(X)[0:5]

# To make an individual prediction using the linear regression model.
print(str(lm.predict(24)))

# Calculate and print Root Mean Square Error(RMSE)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE value: {:.4f}".format(rmse))

# Calculate and print r2_score
from sklearn.metrics import r2_score
print ("R2 Score value: {:.4f}".format(r2_score(y_test, y_pred)))

# Plot the Regression Line
plt.scatter(X, y, color = 'blue', label='Scatter Plot')
plt.plot(X_test, y_pred, color = 'black', linewidth=3, label = 'Regression Line')
plt.title('Relationship between A and B')
plt.xlabel('A')
plt.ylabel('B')
plt.legend(loc=4)
plt.show()

# Plotting residual errors
plt.scatter(lm.predict(X_train), lm.predict(X_train) - y_train, color = 'red', label = 'Train data')
plt.scatter(lm.predict(X_test), lm.predict(X_test) - y_test, color = 'blue', label = 'Test data')
plt.hlines(xmin = 0, xmax = 50, y = 0, linewidth = 3)
plt.title('Residual errors')
plt.legend(loc = 4)
plt.show()

# Checking for Overfitting or Underfitting the data
print("Training set score: {:.4f}".format(lm.score(X_train,y_train)))
print("Test set score: {:.4f}".format(lm.score(X_test,y_test)))

# Save model for future use
from sklearn.externals import joblib
joblib.dump(lm, 'lm_regressor.pkl')
# To load the model
# lm2=joblib.load('lm_regressor.pkl')