# Part 22 - Artificial Neural Network (ANN)

# Import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.layers import Dense
from keras.models import Sequential

print(tf.__version__)

# Import dataset
dataset = pd.read_csv('datasets/Churn_Modelling.csv')

# Separates columns for processing
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values 
print(X)
print(y)

# Encodes categorical data
# Assigns the "Gender" column with a label of 0 or 1
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)

# Transforms column into one-hot encoded format
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialize ANN
ann = Sequential()
# Adding input layer and first hidden layer
ann.add(Dense(units=6, activation='relu'))
# Second hidden layer
ann.add(Dense(units=6, activation='relu'))
# Output layer
ann.add(Dense(units=1, activation='sigmoid'))

# Compile ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the ANN on the training set
ann.fit(X_train, y_train, batch_size=32, epochs=100)

# Predicting the test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
test_predict = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
print(test_predict)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Prints accuracy score
print(accuracy_score(y_test, y_pred))
