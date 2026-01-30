# Part 2 - Linear Regression

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Path to dataset
dataset = pd.read_csv("datasets/Salary_Data.csv")

# Separates columns for processing
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train linear regression model on training set
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predicting test set results
y_pred = lr.predict(X_test)

# Scatter plot to show model results using training set
plt.title("Salary vs Experience (Training Set)")
plt.figure("Salary vs Experience (Training Set)")
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, lr.predict(X_train), color='blue')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")

# Scatter plot to show model results using test set
plt.title("Salary vs Experience (Test Set)")
plt.figure("Salary vs Experience (Test Set)")
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, lr.predict(X_train), color='blue')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")

# Show plots
plt.show()