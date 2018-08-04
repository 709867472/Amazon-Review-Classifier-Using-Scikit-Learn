import csv
import sklearn
import nltk
from nltk.corpus import stopwords
import re
import time
start_time = time.time()


inputFile = open("reviews.csv")
reader = csv.reader(inputFile, delimiter='|')
next(reader)

# get all the stopWords and put them into set
stopWords = set(stopwords.words('english'))

# skip first line
next(reader)
labels, text = [], []
for row in reader:
    labels.append(row[0])
    # split the reviews using characters except alphabetic letters, numbers and single quote
    text.append(re.split("[^a-zA-Z0-9']+", row[1].lower()))

# for each word, we count how many times it appears in positive reviews and how many times it
# appears in negative reviews
goodCount, badCount = {}, {}
for i in range(len(text)):
    if (i + 1) % 5 == 0: continue
    for word in text[i]:
        if word in stopWords: continue
        if labels[i] == "positive":
            if word in goodCount: goodCount.update({word: goodCount[word] + 1})
            else: goodCount.update({word: 1})
        else:
            if word in badCount: badCount.update({word: badCount[word] + 1})
            else: badCount.update({word: 1})

# we assume that for each word, number of times it appears in positive word / total number of
# times it appears in reviews is "goodness". For each review, we sum up the goodness of all to
# get the goodness of the review, if it larger than 0.5, it is a positive review
total, count = 0, 0
for i in range(len(text)):
    if (i + 1) % 5 ==0:
        total += 1
        goodSum, badSum = 0, 0
        for word in text[i]:
            good = goodCount[word] if word in goodCount else 0
            bad = badCount[word] if word in badCount else 0
            if good == 0 and bad == 0: continue
            goodSum += float(good) / (good + bad)
            badSum += float(bad) / (good + bad)
        sentiment = "positive" if goodSum > badSum else "negative"
        if sentiment == labels[i]: count += 1

print(str(float(count) / total) + '\n')

print("--- %s seconds ---" % (time.time() - start_time))
