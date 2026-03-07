# Part 21 - Natural Language Processing

# Import libraries
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Import dataset
dataset = pd.read_csv('datasets/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Downloads stopwords package from nltk library
nltk.download('stopwords')

# Goes through entire dataset and cleans the text
corpus = []
for i in range(0, len(dataset)):
    # Regex to find all non alphabetical characters and replace them with space
    review = re.sub("[^a-zA-Z]", ' ', dataset['Review'][i])
    # Sets all words to lowercase and splits them
    review = review.lower()
    review = review.split()
    # Removes suffixes from words
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    all_stopwords.remove('isn\'t')
    all_stopwords.remove('aren\'t')
    all_stopwords.remove('wasn\'t')
    all_stopwords.remove('weren\'t')
    all_stopwords.remove('don\'t')
    all_stopwords.remove('doesn\'t')
    all_stopwords.remove('didn\'t')
    all_stopwords.remove('won\'t')
    all_stopwords.remove('wouldn\'t')
    all_stopwords.remove('shan\'t')
    all_stopwords.remove('shouldn\'t')
    all_stopwords.remove('couldn\'t')
    all_stopwords.remove('mustn\'t')
    # Adds unqique stopwords to the list
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

print(corpus)

# Creating Bag of Words model
cv = CountVectorizer(max_features=1500)
# Converts text list into tokens for processing
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

print(X)
print(y)

# Splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes model to the training set,
# to predict if the review is positive or negative
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predicting the test set results
y_pred = gnb.predict(X_test)
results = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
print(results)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Prints accuracy score
print(accuracy_score(y_test, y_pred))
