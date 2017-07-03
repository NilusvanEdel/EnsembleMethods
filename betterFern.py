import numpy as np
import csv
import itertools
from dataInformation import DataInformation


class BetterFern:
    def __init__(self, X, Y, no): # X=trainingsSet, Y=trainingsLabel, no=number of splits
        self.X = X
        self.Y = Y
        self.num_of_splits = no
        self.splits = []
        self.split_axis = []
        self.split_vals = []
        for i in range(no):
            self.split_axis.append(np.random.randint(0, X.shape[1]))  # random split axis of all possible axis
            unique_val = np.unique(X[:, self.split_axis])  # random value of all possible split value
            self.split_vals.append(unique_val[np.random.randint(0, len(unique_val))])
        self.lookup_dict, self.lookup_dict_ind = self.split_set(self.X, self.Y)

    def split_set(self, X, Y):
        axis = self.split_axis
        values = self.split_vals
        iter_all = [comb for iters in list(itertools.product([True, False], repeat=i + 1)
                                           for i in range(len(axis))) for comb in iters]
        pos_entries = [s for s in iter_all if len(s) == len(axis)]
        lookup_dict_ind = {el:[] for el in pos_entries}
        lookup_dict = {el:[0, 0] for el in pos_entries}
        for x in range(len(X)):
            tmp = []
            for i in range(len(axis)):
                tmp.append(X[x, axis[i]] == values[i])
            lookup_dict_ind[tuple(tmp)].append(x)
            if Y[x] == 'e':
                lookup_dict[tuple(tmp)][0] += 1
            else:
                lookup_dict[tuple(tmp)][1] += 1
        return lookup_dict, lookup_dict_ind

    def pred(self, X, Y):
        axis = self.split_axis
        values = self.split_vals
        right_pred = 0
        preds = []
        for x in range(len(X)):
            tmp = []
            for i in range(len(axis)):
                tmp.append(X[x, axis[i]] == values[i])
            if self.lookup_dict[tuple(tmp)][0] > self.lookup_dict[tuple(tmp)][1]:
                preds.append('e')
                if Y[x] == 'e':
                    right_pred += 1
            elif self.lookup_dict[tuple(tmp)][0] < self.lookup_dict[tuple(tmp)][1]:
                preds.append('p')
                if Y[x] == 'p':
                    right_pred += 1
        #print(right_pred, " right predictions from ", len(X), " , acc= ", right_pred / len(X))
        return right_pred, preds
'''
di = DataInformation(True)
trials = 200
no_splits = 16
right_pred_all = [0]*no_splits
for i in range(trials):
    print("begin with trials: ", i)
    for l in range(1, no_splits):
        X_train, X_test, y_train, y_test = di.get_TestTrain(0.3)
        fern = BetterFern(X_train, y_train, l)
        right_pred, _ = fern.pred(X_test, y_test)
        right_pred_all[l] += right_pred
        #print(right_pred, " right predictions from ", len(X_test), " , acc= ", right_pred / len(X_test))
for i in range(1, no_splits):
    print("average right predictions with ", i, " splits in ,", trials, " trials: acc= ",
                                                    right_pred_all[i] / (len(X_test)*trials))
'''

