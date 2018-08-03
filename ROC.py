import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import cross_validation
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

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


random_state = np.random.RandomState(0)

probs1 = tree.DecisionTreeClassifier().fit(X_train, trainLabels).fit(X_train, trainLabels).predict_proba(X_test)[:,1]

probs2 = MLPClassifier(solver = 'adam', alpha = 1e-5, hidden_layer_sizes = (5, 2), random_state = 1).fit(X_train, trainLabels).fit(X_train, trainLabels).predict_proba(X_test)[:,1]

probs3 = MultinomialNB().fit(X_train, trainLabels).fit(X_train, trainLabels).predict_proba(X_test)[:,1]

# Compute ROC curve and ROC area for each class
fpr1, tpr1, threshold1 = roc_curve(testLabels, probs1)
fpr2, tpr2, threshold2 = roc_curve(testLabels, probs2)
fpr3, tpr3, threshold3 = roc_curve(testLabels, probs3)

roc_auc1 = auc(fpr1, tpr1)
roc_auc2 = auc(fpr2, tpr2)
roc_auc3 = auc(fpr3, tpr3)

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr1, tpr1, color='green',
         lw=lw, label='Decision Tree (area = %0.2f)' % roc_auc1)
plt.plot(fpr2, tpr2, color='blue',
         lw=lw, label='Neural Network (area = %0.2f)' % roc_auc2)
plt.plot(fpr3, tpr3, color='red',
         lw=lw, label='Naive Bayes (area = %0.2f)' % roc_auc3)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
