import gzip
import json

import matplotlib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt


with gzip.open("../../goemotions.json.gz", "rb") as f:
    fullData = json.loads(f.read().decode("ascii"))

length = len(fullData)
emotions = [None] * length
posts = [None] * length
sentiments = [None] * length

for i in range(length):
    emotions[i] = fullData[i][1]
    posts[i] = fullData[i][0]
    sentiments[i] = fullData[i][2]

print('--------------BASIC NAIVE BAYES EMOTIONS------------------------')
vectorizerBE = CountVectorizer(stop_words='english')
emotions_encoded = emotions
posts_encoded = vectorizerBE.fit_transform(posts)
print("The length of the vocabulary is " + str(len(vectorizerBE.vocabulary_)))
X_trainBE, X_testBE, y_trainBE, y_testBE = train_test_split(posts_encoded, emotions_encoded, stratify=emotions_encoded,
                                                            test_size=0.2, random_state=0)
classifierBE = MultinomialNB()
modelBE = classifierBE.fit(X_trainBE, y_trainBE)
predictions = classifierBE.predict(X_testBE)
f = open("../../performance.txt", "a")
f.write("Bayes Emotions Confusion Matrix")
f.write("\n")
f.write(str(confusion_matrix(y_testBE, predictions)))
f.write("\n")
f.write(str(classification_report(y_testBE, predictions)))
f.write("\n")
f.close()
print("The accuracy of the basic Bayes model for emotions is " + str(
    accuracy_score(classifierBE.predict(X_testBE), y_testBE) * 100))
print('------------------------------------------------------------------')
