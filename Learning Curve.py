import matplotlib.pyplot as plt
import csv
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
import numpy
import time
import re
start_time = time.time()


inputFile = open("reviews.csv")
outputFile = open('/Users/ruichen/Desktop/output.txt', 'w')
reader = csv.reader(inputFile, delimiter = '|')
next(reader)

stopWords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

trainLabels, trainText, testLabels, testText = [], [], [], []
i = 0
for row in reader:
    if (i + 1) % 5 == 0:
        testLabels.append(1 if row[0] == 'positive' else 0)
        testText.append(row[1])
    else:
        trainLabels.append(1 if row[0] == 'positive' else 0)
        trainText.append(row[1])
    i += 1

text = trainText[: 300000]
labels = trainLabels[: 300000]
subText, subLabels, sizes, accuracies_tree, accuracies_net, accuracies_neighbour = [], [], [0], [0.5], [0.5], [0.5]

for i in range(len(text)):
    subText.append(text[i])
    subLabels.append(labels[i])
    if (i + 1) % 20000 == 0:
        sizes.append(i + 1)
        count_vect = CountVectorizer()
        X_train = count_vect.fit_transform(subText)
        X_test = count_vect.transform(testText)

        clf1 = tree.DecisionTreeClassifier().fit(X_train, subLabels)
        accuracy_tree = numpy.sum(clf1.predict(X_test) == testLabels) / float(len(testLabels))
        accuracies_tree.append(accuracy_tree)

        clf2 = MLPClassifier(solver = 'adam', alpha = 1e-5, hidden_layer_sizes = (5, 2), random_state = 1).fit(X_train, subLabels)
        accuracy_net = numpy.sum(clf2.predict(X_test) == testLabels) / float(len(testLabels))
        accuracies_net.append(accuracy_net)

        clf3 = MultinomialNB().fit(X_train, subLabels)
        accuracy_neighbour = numpy.sum(clf3.predict(X_test) == testLabels) / float(len(testLabels))
        accuracies_neighbour.append(accuracy_neighbour)



plt.figure(figsize = (15, 10))
plt.plot(sizes, accuracies_tree, 'g', linewidth = 3)
plt.plot(sizes, accuracies_net,'b:', linewidth = 3)
plt.plot(sizes, accuracies_neighbour, 'r-.', linewidth = 3)
plt.xlabel('size of training data set')
plt.ylabel('accuracy(%)')
plt.legend(['Decision Tree', 'Neural Network', 'Naive Bayes'], loc="lower right")
plt.grid(True)
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()
