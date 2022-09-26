import json
import gzip
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer



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




vectorizer = CountVectorizer()
posts_matrix = vectorizer.fit_transform(posts)










