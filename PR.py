import csv
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
import numpy
import time
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
start_time = time.time()


inputFile = open("reviews.csv")
outputFile = open('/Users/ruichen/Desktop/output.txt', 'w')
reader = csv.reader(inputFile, delimiter = '|')
next(reader)

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

# trainText = trainText[: 5000]
# trainLabels = trainLabels[: 5000]

count_vect = CountVectorizer()
X_train = count_vect.fit_transform(trainText)
X_test = count_vect.transform(testText)

pred_pro1 = tree.DecisionTreeClassifier().fit(X_train, trainLabels).predict_proba(X_test)[:,1]
pred_pro2 = MLPClassifier(solver = 'adam', alpha = 1e-5, hidden_layer_sizes = (5, 2), random_state = 1).fit(X_train, trainLabels).predict_proba(X_test)[:,1]
pred_pro3 = MultinomialNB().fit(X_train, trainLabels).predict_proba(X_test)[:,1]

average_precision1 = average_precision_score(testLabels, pred_pro1)
average_precision2 = average_precision_score(testLabels, pred_pro2)
average_precision3 = average_precision_score(testLabels, pred_pro3)

precision1, recall1, _ = precision_recall_curve(testLabels, pred_pro1)
precision2, recall2, _ = precision_recall_curve(testLabels, pred_pro2)
precision3, recall3, _ = precision_recall_curve(testLabels, pred_pro3)

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(recall1, precision1, color='green',
         lw=lw, label='Decision Tree (AP = %0.2f)' % average_precision1)
plt.plot(recall2, precision2, color='blue',
         lw=lw, label='Neural Network (AP = %0.2f)' % average_precision2)
plt.plot(recall3, precision3, color='black',
         lw=lw, label='Naive Bayes (AP = %0.2f)' % average_precision3)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.title('Precision Recall Curve')
plt.legend(loc="lower right")
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))
