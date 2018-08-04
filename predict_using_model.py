import sklearn
import pickle
import time
start_time = time.time()

outputFile = open('output.txt', 'w')
with open('test_case.txt') as file:
    data = " ".join(line.rstrip("\n").strip() for line in file)
testText = []
testText.append(data)

with open('model.pkl') as model:
    count_vect = pickle.load(model)
    clf = pickle.load(model)

X_train = count_vect.transform(testText)
res = clf.predict(X_train)
outputFile.write(str(res[0]))
print("--- %s seconds ---" % (time.time() - start_time))
