import csv
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
import numpy
import time
start_time = time.time()


inputFile = open("reviews.csv")
reader = csv.reader(inputFile, delimiter='|')
# skip the first line
next(reader)

# get our training and test data
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
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(text_train)
# here we can't use fit_transform or it will throw an error
X_test = count_vect.transform(text_test)

# Decision Tree
# clf = tree.DecisionTreeClassifier()

# Neural Network
# clf = MLPClassifier(solver='adam', alpha=1e-5,
#                     hidden_layer_sizes=(5, 2), random_state=1)

# Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, Y_train)

clf = clf.fit(X_train, Y_train)
accuracy = numpy.sum(clf.predict(X_test) == Y_test) / float(len(Y_test))
print(accuracy)

print("--- %s seconds ---" % (time.time() - start_time))
