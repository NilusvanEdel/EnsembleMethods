import numpy as np
import csv
import itertools
from dataInformation import DataInformation


class Fern:
    def __init__(self, X, Y, no, continuous=False): # X=trainingsSet, Y=trainingsLabel, no=number of splits
        self.X = X
        self.Y = Y
        self.num_of_splits = no
        self.splits = []
        self.split_axis = []
        self.continuous = continuous
        self.split_vals = []
        for i in range(no):
            self.split_axis.append(np.random.randint(0, X.shape[1]))  # random split axis of all possible axis
            unique_val = np.unique(X[:, self.split_axis])  # random value of all possible split value
            if not self.continuous:
                self.split_vals.append(unique_val[np.random.randint(0, len(unique_val))])
            else:
                self.split_vals.append(np.random.uniform(np.float32(unique_val[0]), np.float32(unique_val[-1])))
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
                if not self.continuous:
                    tmp.append(X[x, axis[i]] == values[i])
                else:
                    tmp.append(np.float32(X[x, axis[i]]) <= values[i])
            lookup_dict_ind[tuple(tmp)].append(x)
            if Y[x] == -1:
                lookup_dict[tuple(tmp)][0] += 1
            else:
                lookup_dict[tuple(tmp)][1] += 1
        return lookup_dict, lookup_dict_ind

    def pred(self, X, Y):
        axis = self.split_axis
        values = self.split_vals
        preds = []
        for x in range(len(X)):
            tmp = []
            for i in range(len(axis)):
                if not self.continuous:
                    tmp.append(X[x, axis[i]] == values[i])
                else:
                    tmp.append(np.float32(X[x, axis[i]]) <= values[i])
            if self.lookup_dict[tuple(tmp)][0] > self.lookup_dict[tuple(tmp)][1]:
                preds.append('-1')
            elif self.lookup_dict[tuple(tmp)][0] < self.lookup_dict[tuple(tmp)][1]:
                preds.append('1')
            else:
                if np.random.randint(0,1):
                    preds.append('1')
                else:
                    preds.append('-1')
        preds = np.asarray(preds, dtype=np.float32)
        return preds
