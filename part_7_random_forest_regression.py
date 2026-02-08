# Part 7 - Random Forest Regression

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Path to dataset
dataset = pd.read_csv("datasets/Position_Salaries.csv")

# Separates columns for processing
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training Random Forest Regression model on the whole dataset
forest_regressor = RandomForestRegressor(n_estimators=10, random_state=0)
forest_regressor.fit(X, y)

# Predict a new result
pred_result = forest_regressor.predict([[6.5]])
print(pred_result)

# Visualising the Random Forest Regression results (HD)
X_flat = np.ravel(X)    # Flatten to one element array, fixes DeprecationWarning
X_grid = np.arange(min(X_flat), max(X_flat), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.figure("Truth or Bluff (Random Forest Regression) HD")
plt.scatter(X, y, color="red")
plt.plot(X_grid, forest_regressor.predict(X_grid), color="blue")
plt.title("Truth or Bluff (Random Forest Regression) HD")
plt.xlabel("Position level")
plt.ylabel("Salary")

# Show plots
plt.show()