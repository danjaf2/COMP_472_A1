import json
import gzip

import nltk
from pandas import DataFrame
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import matplotlib
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords




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


# distribution= [sentiments.count('positive')/length*100, sentiments.count('neutral')/length*100, sentiments.count('neutral')/length*100, sentiments.count('ambiguous')/length*100]
# labels = 'Positive', 'Neutral', 'Negative', 'Ambiguous'
# explode = (0, 0.1, 0,0)  # only "explode" the 2nd slice (i.e. 'Hogs')
#
# plt.pie(distribution, explode=explode, labels=labels, autopct='%1.1f%%',
#         shadow=True, startangle=90)
# plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#
# plt.savefig("sentiments.png")
# plt.rcParams.update({'font.size': 3})
#
# plt.hist(emotions, bins=28, alpha=0.5)
# plt.title('Emotions')
# plt.xlabel('X')
# plt.ylabel('Y')
#
# plt.show()

#Basic Naive Bayes

print('--------------BASIC NAIVE BAYES SENTIMENTS------------------------')
vectorizer = CountVectorizer()
sentiments_enconded= sentiments
posts_encoded = vectorizer.fit_transform(posts)
print("The length of the vocabulary is "+str(len(vectorizer.vocabulary_)) )
X_train, X_test, y_train, y_test=train_test_split(posts_encoded,sentiments_enconded, stratify=sentiments_enconded, test_size=0.2, random_state=0)
classifier= MultinomialNB()
modelBE= classifier.fit(X_train, y_train)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
predictions = classifier.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))
print('------------------------------------------------------------------')












