# Multiple Linear Regression Exercise 1

# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Path to dataset
dataset = pd.read_csv("datasets/50_Startups.csv")

# Separates columns for processing
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encode the Independent Variable (State)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training multiple linear regression model on the training set
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predicting the test set results
y_pred = lr.predict(X_test)
# Prints values up to 2 decimals places
np.set_printoptions(precision=2)        
# Reshapes array to be vertical
array_reshaped = np.concatenate((y_pred.reshape(len(y_pred), 1), (y_test.reshape(len(y_test), 1))), axis=1)

sample_pred = [[1, 0, 0, 160000, 130000, 300000]]
prediction = lr.predict(sample_pred)
print("Predicted Profit($):", prediction)

print(lr.coef_)
print(lr.intercept_)