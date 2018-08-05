import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import time
start_time = time.time()

inputFile = open("reviews.csv")
reader = csv.reader(inputFile, delimiter='|')
# skip the fist line
next(reader)

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


# get feature matrix
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(text_train)
X_test = count_vect.transform(text_test)

# Decision Tree
clf_DT = tree.DecisionTreeClassifier().fit(X_train, Y_train)
probs_DT = clf_DT.predict_proba(X_test)[:, 1]

# Neural Network
clf_NN = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2),
                       random_state=1).fit(X_train, Y_train)
probs_NN = clf_NN.predict_proba(X_test)[:, 1]

# Naive Bayes
clf_NB = MultinomialNB().fit(X_train, Y_train)
probs_NB = clf_NB.predict_proba(X_test)[:, 1]

# get false-positive rate and true-positive rate
fpr_DT, tpr_DT, _ = roc_curve(Y_test, probs_DT)
fpr_NN, tpr_NN, _ = roc_curve(Y_test, probs_NN)
fpr_NB, tpr_NB, _ = roc_curve(Y_test, probs_NB)

# get area under curve
roc_auc_DT = auc(fpr_DT, tpr_DT)
roc_auc_NN = auc(fpr_NN, tpr_NN)
roc_auc_NB = auc(fpr_NB, tpr_NB)

# pot curve
plt.figure()
lw = 2
plt.figure(figsize=(10, 10))
plt.plot(fpr_DT, tpr_DT, color='green',
         lw=lw, label='Decision Tree (area = %0.2f)' % roc_auc_DT)
plt.plot(fpr_NN, tpr_NN, color='blue',
         lw=lw, label='Neural Network (area = %0.2f)' % roc_auc_NN)
plt.plot(fpr_NB, tpr_NB, color='red',
         lw=lw, label='Naive Bayes (area = %0.2f)' % roc_auc_NB)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
plt.title('ROC Curve', fontsize=20)
plt.legend(loc="lower right")
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()
