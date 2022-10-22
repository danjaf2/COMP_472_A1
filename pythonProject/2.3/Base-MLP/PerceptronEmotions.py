import gzip
import json

import matplotlib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

matplotlib.use('TkAgg')
# from nltk.corpus import stopwords


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

print('--------------PERCEPTRON EMOTIONS------------------------')
vectorizer = CountVectorizer(stop_words='english')
posts_encoded = vectorizer.fit_transform(posts)
print("The length of the vocabulary is " + str(len(vectorizer.vocabulary_)))
X_train, X_test, y_train, y_test = train_test_split(posts_encoded, emotions, stratify=emotions, test_size=0.2,
                                                    random_state=0)
classifier = MLPClassifier(max_iter=1)
modelBE = classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
f = open("../../performance.txt", "a")
f.write("Perceptron Emotions Confusion Matrix")
f.write("\n")
f.write(str(confusion_matrix(y_test, predictions)))
f.write("\n")
f.write(str(classification_report(y_test, predictions)))
f.write("\n")
f.close()
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))
print('------------------------------------------------------------------')
