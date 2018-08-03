import csv
import sklearn
import re
import time
start_time = time.time()


inputFile = open("reviews.csv")
outputFile = open('/Users/ruichen/Desktop/output.txt', 'w')
reader = csv.reader(inputFile, delimiter = '|')
next(reader)

inputFile = open("reviews.csv")
outputFile = open('/Users/ruichen/Desktop/output.txt', 'w')

stopWords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

reader = csv.reader(inputFile, delimiter = '|')
# skip first line
next(reader)
label = []
text = []
for row in reader:
    label.append(row[0])
    text.append(re.split("[^a-zA-Z0-9']+", row[1]))
    # text.append(row[1].split(' '))

# goodCount = {}
# badCount = {}
# for i in range(len(text)):
#     if (i + 1) % 5 == 0: continue
#     for word in text[i]:
#         if word in stopWords: continue
#         if label[i] == "positive":
#             if word in goodCount: goodCount.update({word: goodCount[word] + 1})
#             else: goodCount.update({word: 1})
#         else:
#             if word in badCount: badCount.update({word: badCount[word] + 1})
#             else: badCount.update({word: 1})

# total = 0
# count = 0
# for i in range(len(text)):
#     if (i + 1) % 5 ==0:
#         total += 1
#         goodSum = 0
#         badSum = 0
#         for word in text[i]:
#             good = goodCount[word] if word in goodCount else 0
#             bad = badCount[word] if word in badCount else 0
#             if good == 0 and bad == 0: continue
#             goodSum += (good + 0.0) / (good + bad)
#             badSum += (bad + 0.0) / (good + bad)
#         sentiment = "positive" if goodSum > badSum else "negative"
#         if sentiment == label[i]: count += 1

goodCount = {}
badCount = {}
for i in range(len(text)):
    for word in text[i]:
        if word in stopWords: continue
        if label[i] == "positive":
            if word in goodCount: goodCount.update({word: goodCount[word] + 1})
            else: goodCount.update({word: 1})
        else:
            if word in badCount: badCount.update({word: badCount[word] + 1})
            else: badCount.update({word: 1})

total = 0
count = 0
for t in text:
    for word in t:
        total += 1
        good = goodCount[word] if word in goodCount else 0
        bad = badCount[word] if word in badCount else 0
        if good > bad: count += 1

print(str(count) + '\n')
print(str(total - count) + '\n')

print("--- %s seconds ---" % (time.time() - start_time))
