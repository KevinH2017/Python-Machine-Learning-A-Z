# Part 29 - XGBoost

# XGBoost (eXtreme Gradient Boosting) is an optimized gradient boosting algorithm that combines multiple
# weaker models into a stronger, higher-performin model. It uses decision trees as base learners to build
# them sequentially to correct errors from previous trees. It also uses parallel processing for faster
# training on larger datasets and allows parameter custimization to improve performance for specific problems.

# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Import dataset
dataset = pd.read_csv('datasets/XGBoost_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Encodes categorical data
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Feature Scaling using XGBoost
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

# Apply K-Fold Cross Validation
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
