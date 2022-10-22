import json
import gzip
import gensim.downloader as gsm
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = gsm.load('word2vec-google-news-300')
with gzip.open("../goemotions.json.gz", "rb") as f:
    fullData = json.loads(f.read().decode("ascii"))

length = len(fullData)
emotions = [None] * length
posts = [None] * length
sentiments = [None] * length

for i in range(length):
    emotions[i] = fullData[i][1]
    posts[i] = fullData[i][0]
    sentiments[i] = fullData[i][2]

# Basic Naive Bayes

print('--------------Word2Vec EMBEDDINGS Top MLP Sentiments------------------------')
vectorizer = CountVectorizer(stop_words='english')
sentiments_encoded = sentiments
posts_encoded = [None] * (len(posts))
for i in range(len(posts)):
    posts_encoded[i] = word_tokenize(posts[i])

X_train, X_test, y_train, y_test = train_test_split(posts_encoded, sentiments_encoded, stratify=sentiments_encoded,
                                                    test_size=0.2, random_state=0)

numberOfTokens = 0;
for i in range(len(X_train)):
    numberOfTokens += len(X_train[i])

print("Number of tokens in training set:")
print(numberOfTokens)

numberOfTokensTest = 0;
for i in range(len(X_test)):
    numberOfTokensTest += len(X_test[i])

print("Number of tokens in testing set:")
print(numberOfTokensTest)

X_train_final = []
counter = 0
numberOfHitsTraining = 0
for arr in X_train:
    posts_encoded_sum = []
    loop = 0
    for i in arr:
        try:
            posts_encoded_sum.append(np.array(data[str(i).lower()]).tolist())
            posts_encoded_sum = np.array(posts_encoded_sum).tolist()
            counter = counter + 1
            numberOfHitsTraining = numberOfHitsTraining + 1
        except:
            counter = counter
    if counter > 0:
        average = np.mean(posts_encoded_sum, axis=0)
        X_train_final.append(average)
        counter = 0
        loop = loop + 1
    else:
        X_train_final.append(np.zeros_like(data['king']).tolist())
        counter = 0
        loop = loop + 1
print("Hit rate percentage of training set is:")
print((numberOfHitsTraining / numberOfTokens) * 100)
X_test_final = []
counter = 0
numberOfHitsTesting = 0
for arr in X_test:
    posts_encoded_sum = []
    loop = 0
    for i in arr:
        try:
            posts_encoded_sum.append(np.array(data[str(i).lower()]).tolist())
            posts_encoded_sum = np.array(posts_encoded_sum).tolist()
            counter = counter + 1
            numberOfHitsTesting = numberOfHitsTesting + 1
        except:
            counter = counter
    if counter > 0:
        average = np.mean(posts_encoded_sum, axis=0)
        X_test_final.append(average)
        counter = 0
        loop = loop + 1
    else:
        X_test_final.append(np.zeros_like(data['king']).tolist())
        counter = 0
        loop = loop + 1

print("Hit rate percentage of training set is:")
print((numberOfHitsTesting / numberOfTokensTest) * 100)

print("Creating model")
params = {
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['adam', 'sgd'],
    'hidden_layer_sizes': [(30, 50), (10, 10, 10)]
}
model_grid = GridSearchCV(estimator=MLPClassifier(max_iter=10), param_grid=params, n_jobs=-1)
model_grid.fit(X_train_final, y_train)

predictions = model_grid.predict(X_test_final)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))
