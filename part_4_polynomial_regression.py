# Part 4 - Polynomial Regression

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Path to dataset
dataset = pd.read_csv("datasets/Position_Salaries.csv")

# Separates columns for processing
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training multiple linear regression model on the whole dataset
lr = LinearRegression()
lr.fit(X, y)

# Training polynomial regression model on the whole dataset
# Higher degree creates higher resolution graph and smoother curve
pf = PolynomialFeatures(degree=4)
X_poly = pf.fit_transform(X)
# Training new linear regression model using the polynomial array
lr_2 = LinearRegression()
lr_2.fit(X_poly, y)

# Visualize the linear regression results
plt.figure("Truth or Bluff (Linear Regression)")
plt.scatter(X, y, color="red")
plt.plot(X, lr.predict(X), color="blue")
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")

# Visualize the polynomial regression results
plt.figure("Truth or Bluff (Polynomial Regression)")
plt.scatter(X, y, color="red")
plt.plot(X, lr_2.predict(X_poly), color="blue")
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")

# Visualize the polynomial regression results for higher resolution and smoother curve
X_flat = np.ravel(X)    # Flatten to one element array, fixes DeprecationWarning
X_grid = np.arange(min(X_flat), max(X_flat), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.figure("Truth or Bluff (Polynomial Regression HD)")
plt.scatter(X, y, color="red")
plt.plot(X_grid, lr_2.predict(pf.fit_transform(X_grid)), color="blue")
plt.title("Truth or Bluff (Polynomial Regression HD)")
plt.xlabel("Position level")
plt.ylabel("Salary")

# Predicting new result with linear regression
print(lr.predict([[6.5]]))

# Predicting new result with polynomial regression
print(lr_2.predict(pf.fit_transform([[6.5]])))

# Show plots
plt.show()