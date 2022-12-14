import gzip
import json

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

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
