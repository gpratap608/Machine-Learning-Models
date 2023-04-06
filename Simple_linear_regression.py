# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
print("real salary")
print(y_test)
print("predicted salary")
print(y_pred)
# Visualising the Training set results
plt.scatter(X_train,y_train,color = 'green')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.title("Experience vs Salary Graph")
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.show()

# Visualising the Test set results
plt.scatter(X_test,y_test,color = 'green')
plt.plot(X_test,regressor.predict(X_test),color = 'blue')
plt.title("Test Experience vs Salary Graph")
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.show()

#predicting a new single value
print(regressor.predict([[1]]))

#getting the coefficient and th intercept value

print(regressor.coef_)
print(regressor.intercept_)