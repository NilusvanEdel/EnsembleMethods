import numpy as np
from betterFern import BetterFern
from sklearn.linear_model import Perceptron
from dataInformation import DataInformation
from sklearn.preprocessing import LabelBinarizer

di = DataInformation(True)
X_train, X_test, y_train, y_test = di.get_TestTrain(0.3)
ferns = []
numberOfFerns = 15
numberOfSplits = 5
fernPreds = []
fernAccuracies = []
for i in range(numberOfFerns):
    tmp = BetterFern(X_train,y_train, numberOfSplits)
    ferns.append(tmp)
    right_pred, preds = tmp.pred(X_test, y_test)
    print('acc in fern ', i, ": ",right_pred/len(X_test))
    fernPreds.append(np.asarray(preds))
    fernAccuracies.append(right_pred/len(X_train))
fernPreds = np.asarray(fernPreds)

X = fernPreds[:,:]=='e'
X = X.astype(float)
y = y_test[:]=='e'
y = y.astype(float)
X = np.transpose(X)
# linear added (average of all weight for all = 1)
rightPred = 0
for i in range(len(X)):
    if sum(X[i]) > numberOfSplits/2 and y[i]==1:
        rightPred += 1
    elif sum(X[i]) < numberOfSplits/2 and y[i]==0:
        rightPred += 1
print("Accuracy for simple linear combination: ", rightPred/len(X))

# linear combination weighted with its accuracies
rightPred = 0
for i in range(len(X)):
    tmpTrue = 0
    counterTrue = 0
    tmpFalse = 0
    counterFalse = 0
    for l in range(len(X[i])):
        if X[i][l] == 1:
            tmpTrue += fernAccuracies[l]
            counterTrue += 1
        else:
            tmpFalse += fernAccuracies[l]
            counterFalse += 1
    if counterTrue != 0: tmpTrue /= counterTrue
    if counterFalse != 0: tmpFalse /= counterFalse

    if tmpTrue > tmpFalse and y[i]==1:
        rightPred += 1
    elif tmpTrue < tmpFalse and y[i]==0:
        rightPred += 1
print("Accuracy for weighted linear combination: ", rightPred / len(X))

# single layer perceptron
clf = Perceptron()
clf.fit(X, y)

print('Accuracy for perceptron:', clf.score(X, y))
