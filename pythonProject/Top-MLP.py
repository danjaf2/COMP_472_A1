import json
import gzip
from pandas import DataFrame
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
from collections import Counter


with gzip.open("goemotions.json.gz", "rb") as f:
    fullData = json.loads(f.read().decode("ascii"))


length = len(fullData)
emotions = [None]*length
posts = [None]*length
sentiments = [None]*length



for i in range(length):
    emotions[i]=fullData[i][1]
    posts[i]=fullData[i][0]
    sentiments[i]= fullData[i][2]


print('--------------BETTER PERFORMING DT------------------------')
vectorizer = CountVectorizer(stop_words='english')
sentiments_encoded = sentiments
emotions_encoded = emotions
posts_encoded = vectorizer.fit_transform(posts)
print("The length of the vocabulary is "+str(len(vectorizer.vocabulary_)))

print('--------------SENTIMENTS------------------------')
X_trainS, X_testS, y_trainS, y_testS = train_test_split(posts_encoded, sentiments_encoded, stratify=sentiments_encoded, test_size=0.2, random_state=0)
params = {
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['adam', 'sgd'],
    'hidden_layer_sizes': [(30, 50), (10,10,10)]
}
model_grid = GridSearchCV(estimator=MLPClassifier(max_iter=10), param_grid=params, n_jobs=-1)
model_grid.fit(X_trainS, y_trainS)
print("The accuracy of the better performing Decision Tree model for sentiments is " + str(accuracy_score(model_grid.predict(X_testS), y_testS)*100))
print(model_grid.best_params_)

# print('--------------EMOTIONS------------------------')
# X_trainE, X_testE, y_trainE, y_testE = train_test_split(posts_encoded, emotions_encoded, stratify=emotions_encoded, test_size=0.2, random_state=0)
# params = {
#     'activation': ['identity', 'logistic', 'tanh', 'relu'],
#     'solver': ['adam', 'sgd'],
#     'hidden_layer_sizes': [(30, 50), (10,10,10)]
# }
# model_grid = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=params)
# model_grid.fit(X_trainE, y_trainE)
# print("The accuracy of the better performing Decision Tree model for emotions is " + str(accuracy_score(model_grid.predict(X_testE), y_testE)*100))
# print('------------------------------------------------------------------')
