import gzip
import json

import matplotlib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier

matplotlib.use('TkAgg')

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

print('--------------BETTER PERFORMING DT------------------------')
vectorizer = CountVectorizer(stop_words='english')
sentiments_encoded = sentiments
emotions_encoded = emotions
posts_encoded = vectorizer.fit_transform(posts)
print("The length of the vocabulary is " + str(len(vectorizer.vocabulary_)))

print('--------------SENTIMENTS------------------------')
X_train, X_test, y_train, y_test = train_test_split(posts_encoded, sentiments_encoded, stratify=sentiments_encoded,
                                                    test_size=0.2, random_state=0)
params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [30, 60],
    'min_samples_split': [2, 4, 8]
}
model_grid = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=params, n_jobs=-1)
model_grid.fit(X_train, y_train)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

predictions = model_grid.predict(X_test)

f = open("../performance.txt", "a")
f.write("TOP Decision Tree Setiments Confusion Matrix")
f.write("\n")
f.write(str(confusion_matrix(y_test, predictions)))
f.write("\n")
f.write(str(classification_report(y_test, predictions)))
f.write("\n")
f.write("Best Estimator: " + str(model_grid.best_params_))
f.write("\n")
f.write("\n")
f.close()

print(model_grid.best_estimator_)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))

print('--------------EMOTIONS------------------------')
X_trainE, X_testE, y_trainE, y_testE = train_test_split(posts_encoded, emotions_encoded, stratify=emotions_encoded,
                                                        test_size=0.2, random_state=0)
params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [30, 60],
    'min_samples_split': [2, 4, 8]
}
model_grid = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=params, n_jobs=-1)
model_grid.fit(X_trainE, y_trainE)

predictions = model_grid.predict(X_testE)
f = open("../performance.txt", "a")
f.write("TOP Decision Tree Emotions Confusion Matrix")
f.write("\n")
f.write(str(confusion_matrix(y_testE, predictions)))
f.write("\n")
f.write(str(classification_report(y_testE, predictions)))
f.write("\n")
f.write("Best Estimator: " + str(model_grid.best_params_))
f.write("\n")
f.close()

print("The accuracy of the better performing Decision Tree model for emotions is " + str(
    accuracy_score(model_grid.predict(X_testE), y_testE) * 100))
print('------------------------------------------------------------------')
