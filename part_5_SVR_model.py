# Part 5 - SVR Intuition

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Path to dataset
dataset = pd.read_csv("datasets/Position_Salaries.csv")

# Separates columns for processing
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Reshapes y to a 2D array
y = y.reshape(len(y), 1)

# Feature Scaling, separately scaling X and y
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Training the SVR model on the whole dataset
regressor = SVR(kernel='rbf')       # Using the Radial Basis Function
regressor.fit(X, y.ravel())         # ravel() to avoid DataConversionWarning error

# Predicting a new result
pred_result = sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1, 1))
print(pred_result)

# Inverse transforming X for visualization
sc_inverse_X = sc_X.inverse_transform(X)
sc_inverse_y = sc_y.inverse_transform(y)

# Visualizing the SVR results
plt.figure("Truth or Bluff (SVR)")
plt.scatter(sc_inverse_X, sc_inverse_y, color="red")
plt.plot(sc_inverse_X, sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)), color="blue")
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")

# Visualizing the SVR results for higher resolution and smoother curve
X_flat = np.ravel(sc_inverse_X)    # Flatten to one element array, fixes DeprecationWarning
X_grid = np.arange(min(X_flat), max(X_flat), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.figure("Truth or Bluff (SVR) HD")
plt.scatter(sc_inverse_X, sc_inverse_y, color="red")
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1, 1)), color="blue")
plt.title("Truth or Bluff (SVR) HD")
plt.xlabel("Position level")
plt.ylabel("Salary")

# Show plots
plt.show()