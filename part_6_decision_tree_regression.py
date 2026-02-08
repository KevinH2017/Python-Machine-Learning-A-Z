# Part 6 - Decision Tree Regression

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Path to dataset
dataset = pd.read_csv("datasets/Position_Salaries.csv")

# Separates columns for processing
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training Decision Tree Regression model on the whole dataset
tree_regressor = DecisionTreeRegressor(random_state=0)
tree_regressor.fit(X, y)

# Predicting a new result
pred_result = tree_regressor.predict([[6.5]])
print(pred_result)

# Visualizing the Decision Tree Regression results
X_flat = np.ravel(X)    # Flatten to one element array, fixes DeprecationWarning
X_grid = np.arange(min(X_flat), max(X_flat), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.figure("Truth or Bluff (Decision Tree Regression) HD")
plt.scatter(X, y, color="red")
plt.plot(X_grid, tree_regressor.predict(X_grid), color="blue")
plt.title("Truth or Bluff (Decision Tree Regression) HD")
plt.xlabel("Position level")
plt.ylabel("Salary")

# Show plots
plt.show()