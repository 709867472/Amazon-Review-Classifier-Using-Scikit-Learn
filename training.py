import csv
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
import numpy
import pickle


inputFile = open("reviews.csv")
outputFile = open('model.pkl', 'w')
reader = csv.reader(inputFile, delimiter='|')
# skip the first line
next(reader)

# get training and test data
Y_train, text_train = [], []
i = 0
for row in reader:
    if (i + 1) % 5 != 0:
        Y_train.append(1 if row[0] == 'positive' else 0)
        text_train.append(row[1])
    i += 1

count_vect = CountVectorizer()
X_train = count_vect.fit_transform(text_train)

# use Neural Newwork to produce the prediction model
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2),
                    random_state=1)
clf.fit(X_train, Y_train)
pickle.dump(count_vect, outputFile)
pickle.dump(clf, outputFile)
