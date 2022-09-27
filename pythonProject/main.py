import json
import gzip
from pandas import DataFrame
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import matplotlib
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
vectorizer = CountVectorizer()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
sentiments_encoded= le.fit_transform(sentiments)
posts_encoded = vectorizer.fit_transform(posts)
print("The length of the vocabulary is "+str(len(vectorizer.vocabulary_)) )

Data = {'Post': posts,'Emotions':emotions, 'Sentiments': sentiments}
df = DataFrame(Data,columns=['Post','Emotions', 'Sentiments'])

classifier = MultinomialNB(class_prior=[0.293, 0.293, 0.313, 0.101])
model= classifier.fit(posts_encoded, sentiments_encoded)
topredict =vectorizer.transform(['Man I love reddit.'])
prediction= model.predict(topredict)
print(prediction)

for i in range (50):
    print(posts[i])
    print(sentiments_encoded[i])
    print('\n')








