import csv
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import numpy
import matplotlib.pyplot as plt
import time
start_time = time.time()


inputFile = open("reviews.csv")
reader = csv.reader(inputFile, delimiter='|')
# skip the first line
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
pred_pro_DT = clf_DT.predict_proba(X_test)[:, 1]

# Neural Network
clf_NN = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2),
                       random_state=1).fit(X_train, Y_train)
pred_pro_NN = clf_NN.predict_proba(X_test)[:, 1]

# Naive Bayes
clf_NB = MultinomialNB().fit(X_train, Y_train)
pred_pro_NB = clf_NB.predict_proba(X_test)[:, 1]

# get average precision
ap_DT = average_precision_score(Y_test, pred_pro_DT)
ap_NN = average_precision_score(Y_test, pred_pro_NN)
ap_NB = average_precision_score(Y_test, pred_pro_NB)

# get precision and recall
precision_DT, recall_DT, _ = precision_recall_curve(Y_test, pred_pro_DT)
precision_NN, recall_NN, _ = precision_recall_curve(Y_test, pred_pro_NN)
precision_NB, recall_NB, _ = precision_recall_curve(Y_test, pred_pro_NB)

# plot curve
plt.figure()
lw = 2
plt.figure(figsize=(10, 10))
plt.plot(recall_DT, precision_DT, color='green',
         lw=lw, label='Decision Tree (AP = %0.2f)' % ap_DT)
plt.plot(recall_NN, precision_NN, color='blue',
         lw=lw, label='Neural Network (AP = %0.2f)' % ap_NN)
plt.plot(recall_NB, precision_NB, color='red',
         lw=lw, label='Naive Bayes (AP = %0.2f)' % ap_NB)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.title('Precision Recall Curve')
plt.legend(loc="lower right")
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))
