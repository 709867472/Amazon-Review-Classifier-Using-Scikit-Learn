import csv
import sklearn
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
import numpy
import time
import matplotlib.pyplot as plt
start_time = time.time()


inputFile = open("reviews.csv")
reader = csv.reader(inputFile, delimiter='|')
# skip the first line
next(reader)

# get all the stopWords and put them into set
stopWords = set(stopwords.words('english'))

# get training and test data
Y_train, text_train, Y_test, text_test = [], [], [], []
i = 0
for row in reader:
    if (i + 1) % 5 == 0:
        Y_test.append(1 if row[0] == 'positive' else 0)
        text_test.append(row[1])
    else:
        Y_train.append(1 if row[0] == 'positive' else 0)
        text_train.append(row[1])
    i += 1

# pick the first 300k training data
text, labels = text_train[: 300000], Y_train[: 300000]
subText, subLabels = [], []
sizes, accs_DT, accs_NN, accs_NB = [0], [0.5], [0.5], [0.5]

# get the accuracy when the training data size is 10k, 20k...300k
for i in range(len(text)):
    subText.append(text[i])
    subLabels.append(labels[i])
    if (i + 1) % 10000 == 0:
        sizes.append((i + 1) / 1000)
        count_vect = CountVectorizer()
        X_train = count_vect.fit_transform(subText)
        X_test = count_vect.transform(text_test)

        clf_DT = tree.DecisionTreeClassifier()
        clf_DT.fit(X_train, subLabels)
        acc_DT = numpy.sum(clf_DT.predict(X_test) == Y_test) / float(len(Y_test))
        accs_DT.append(acc_DT)

        clf_NN = MLPClassifier(solver='adam', alpha=1e-5,
                               hidden_layer_sizes=(5, 2), random_state=1)
        clf_NN.fit(X_train, subLabels)
        acc_NN = numpy.sum(clf_NN.predict(X_test) == Y_test) / float(len(Y_test))
        accs_NN.append(acc_NN)

        clf_NB = MultinomialNB()
        clf_NB.fit(X_train, subLabels)
        acc_NB = numpy.sum(clf_NB.predict(X_test) == Y_test) / float(len(Y_test))
        accs_NB.append(acc_NB)


# draw the Learning Curve to show the relationship between training data
# size and accuracy
plt.figure(figsize=(15, 10))
plt.plot(sizes, accs_DT, 'g', linewidth=3)
plt.plot(sizes, accs_NN, 'b:', linewidth=3)
plt.plot(sizes, accs_NB, 'r-.', linewidth=3)
plt.xlabel('Size of Training Data(K)', fontsize=15)
plt.ylabel('Accuracy(%)', fontsize=15)
plt.title('Learning Curve', fontsize=20)
plt.legend(['Decision Tree', 'Neural Network', 'Naive Bayes'],
           loc="lower right")
plt.grid(True)
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()
