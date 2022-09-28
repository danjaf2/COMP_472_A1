import json
import gzip
from pandas import DataFrame
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import matplotlib
from sklearn import preprocessing
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
vectorizerBS = CountVectorizer()
leBS = preprocessing.LabelEncoder()
sentiments_encoded= leBS.fit_transform(sentiments)
posts_encoded = vectorizerBS.fit_transform(posts)
print("The length of the vocabulary is "+str(len(vectorizerBS.vocabulary_)) )
X_trainBS, X_testBS, y_trainBS, y_testBS=train_test_split(posts_encoded,sentiments_encoded, stratify=sentiments_encoded, test_size=0.2, random_state=0)
classifierBS = MultinomialNB()
modelBS= classifierBS.fit(X_trainBS, y_trainBS)
from sklearn.metrics import accuracy_score
print("The accuracy of the basic Bayes model for sentiments is " + str(accuracy_score(classifierBS.predict(X_testBS),y_testBS)))
print('------------------------------------------------------------------')
print('--------------BASIC NAIVE BAYES EMOTIONS------------------------')
vectorizerBE = CountVectorizer()
leBE = preprocessing.LabelEncoder()
emotions_encoded= leBE.fit_transform(emotions)
posts_encoded = vectorizerBE.fit_transform(posts)
print("The length of the vocabulary is "+str(len(vectorizerBE.vocabulary_)) )
X_trainBE, X_testBE, y_trainBE, y_testBE=train_test_split(posts_encoded,emotions_encoded, stratify=emotions_encoded, test_size=0.2, random_state=0)
classifierBE= MultinomialNB()
modelBE= classifierBE.fit(X_trainBE, y_trainBE)
from sklearn.metrics import accuracy_score
print("The accuracy of the basic Bayes model for emotions is " + str(accuracy_score(classifierBE.predict(X_testBE),y_testBE)))
print('------------------------------------------------------------------')











