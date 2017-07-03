import numpy as np
from betterFern import BetterFern
from sklearn.linear_model import Perceptron
from dataInformation import DataInformation
from sklearn.preprocessing import LabelBinarizer
import cDTLearner as cDTL

di = DataInformation(True)
ferns = []
trials = 50
numberOfSplits = 5
acc_list = []
for numberOfFerns in range(2,10):
    accuracies = [0, 0, 0, 0]
    for trial in range(trials):
        print("begin trial: ", trial)
        X_train, X_test, X_val, y_train, y_test, y_val = di.get_TestTrainVal(0.3, 0.33, True)


        fernPreds_test = []
        fernPreds_val = []
        fernAccuracies = []
        for i in range(numberOfFerns):
            tmp = BetterFern(X_train,y_train, numberOfSplits)
            ferns.append(tmp)
            right_pred_X_test, preds_test = tmp.pred(X_test, y_test)
            right_pred_x_val, preds_val = tmp.pred(X_val, y_val)
            #print('acc in fern ', i, ": ",right_pred_x_val/len(X_val))
            fernPreds_test.append(np.asarray(preds_test))
            fernPreds_val.append(np.asarray(preds_val))
            fernAccuracies.append(right_pred_X_test/len(X_test))
        fernPreds_test = np.asarray(fernPreds_test)
        fernPreds_val = np.asarray(fernPreds_val)

        # transform into numerical with right shape
        X_pred_test = np.asarray([f == 'e' for f in fernPreds_test])+0
        X_pred_test = np.transpose(X_pred_test)
        X_pred_val = np.asarray([f == 'e' for f in fernPreds_val])+0
        X_pred_val = np.transpose(X_pred_val)
        y_test = np.asarray(y_test[:]=='e')+0
        y_val = np.asarray(y_val=='e')+0
        # majority vote
        rightPred_lin = 0
        for i in range(len(X_pred_val)):
            if sum(X_pred_val[i]) >= numberOfSplits/2.0 and y_val[i]==1:
                rightPred_lin += 1
            elif sum(X_pred_val[i]) < numberOfSplits/2.0 and y_val[i]==0:
                rightPred_lin += 1
        acc_majVote = rightPred_lin/len(X_pred_val)
        accuracies[0] += acc_majVote
        #print("Accuracy for majority vote: ", acc_majVote)

        # majority vote with weights
        rightPred_wei = 0
        for i in range(len(X_pred_val)):
            tmpTrue = 0
            counterTrue = 0
            tmpFalse = 0
            counterFalse = 0
            for l in range(numberOfFerns):
                if X_pred_val[i][l] == 1:
                    tmpTrue += fernAccuracies[l]
                    counterTrue += 1
                else:
                    tmpFalse += fernAccuracies[l]
                    counterFalse += 1
            if counterTrue != 0: tmpTrue /= counterTrue
            if counterFalse != 0: tmpFalse /= counterFalse

            if tmpTrue > tmpFalse and y_val[i]==1:
                rightPred_wei += 1
            elif tmpTrue < tmpFalse and y_val[i]==0:
                rightPred_wei += 1
        acc_majVoteWei = rightPred_wei / len(X_pred_val)
        accuracies[1] += acc_majVoteWei
        #print("Accuracy for weighted majority vote: ", acc_majVoteWei)


        # single layer perceptron
        clf = Perceptron()
        clf.fit(X_test, y_test)

        acc_Perceptron = clf.score(X_val, y_val)
        accuracies[2] += acc_Perceptron
        #print('Accuracy for perceptron:', acc_Perceptron)

        ensemble_tree = cDTL.Learner(max_depth = 4)
        ensemble_tree.learn(X_test,y_test,['d']*X_test.shape[1])


        Y_hat = np.array([ensemble_tree.predict(x) for x in X_val])
        acc_tree = np.mean(Y_hat==y_val)
        accuracies[3] += acc_tree
        #print('Accuracy for tree classifier: {0}'.format(acc_tree))

    for i in range(len(accuracies)):
        accuracies[i] /= trials
        if i == 0:
            print("Average accuracy for maj vote in ", trials,
                  " with", numberOfFerns, "number of ferns: ", accuracies[i])
        elif i == 1:
            print("Average accuracy for weighted maj vote in ", trials,
                  " with", numberOfFerns, "number of ferns: ", accuracies[i])
        elif i == 2:
            print("Average accuracy for perceptron in ", trials,
                  " with", numberOfFerns, "number of ferns: ", accuracies[i])
        elif i == 3:
            print("Average accuracy for decision tree in ", trials,
                  " with", numberOfFerns, "number of ferns: ", accuracies[i])
    acc_list.append(accuracies)

print("over")
