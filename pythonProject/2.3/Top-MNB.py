import json
import gzip
from pandas import DataFrame
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
from collections import Counter
from sklearn.model_selection import cross_validate

with gzip.open("../goemotions.json.gz", "rb") as f:
    fullData = json.loads(f.read().decode("ascii"))


length = len(fullData)
emotions = [None]*length
posts = [None]*length
sentiments = [None]*length



for i in range(length):
    emotions[i]=fullData[i][1]
    posts[i]=fullData[i][0]
    sentiments[i]= fullData[i][2]


print('--------------BETTER PERFORMING MNB------------------------')
vectorizer = CountVectorizer(stop_words='english')
sentiments_encoded = sentiments
emotions_encoded = emotions
posts_encoded = vectorizer.fit_transform(posts)
print("The length of the vocabulary is "+str(len(vectorizer.vocabulary_)))

print('--------------SENTIMENTS------------------------')
X_trainS, X_testS, y_trainS, y_testS = train_test_split(posts_encoded, sentiments_encoded, stratify=sentiments_encoded, test_size=0.2, random_state=0)
params = [
{'alpha': [0,0.5,3/4,0.8]}
]

model_grid = GridSearchCV(estimator=MultinomialNB(), param_grid=params, n_jobs=-1)
model_grid.fit(X_trainS, y_trainS)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
predictions = model_grid.predict(X_testS)
f = open("../performance.txt", "a")
f.write("TOP Bayes Sentiment Confusion Matrix")
f.write("\n")
f.write(str(confusion_matrix(y_testS,predictions)))
f.write("\n")
f.write(str(classification_report(y_testS,predictions)))
f.write("\n")
f.write("Best Estimator: "+str(model_grid.best_params_))
f.write("\n")
f.write("\n")
f.close()

print("The accuracy of the better performing Multinomial Naive Bayes Classifier model for sentiments is " + str(accuracy_score(model_grid.predict(X_testS), y_testS)*100))
print(model_grid.best_estimator_)
print('--------------EMOTIONS------------------------')
X_trainE, X_testE, y_trainE, y_testE = train_test_split(posts_encoded, emotions_encoded, stratify=emotions_encoded, test_size=0.2, random_state=0)
params = {
    'alpha': [0, 0.5,0.45,0.5145]
}
model_grid = GridSearchCV(estimator=MultinomialNB(), param_grid=params, n_jobs=-1)
model_grid.fit(X_trainE, y_trainE)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
predictions = model_grid.predict(X_testE)
f = open("../performance.txt", "a")
f.write("TOP Bayes Emotions Confusion Matrix")
f.write("\n")
f.write(str(confusion_matrix(y_testE,predictions)))
f.write("\n")
f.write(str(classification_report(y_testE,predictions)))
f.write("\n")
f.write("Best Estimator: "+str(model_grid.best_params_))
f.write("\n")
f.close()

print("The accuracy of the better performing Multinomial Naive Bayes Classifier model for emotions is " + str(accuracy_score(model_grid.predict(X_testE), y_testE)*100))
print(model_grid.best_estimator_)
print('------------------------------------------------------------------')
