import numpy as np
import csv
import itertools


class Fern:
    def __init__(self, X, Y, no):
        self.X = X
        self.Y = Y
        self.num_of_splits = no
        self.splits = []
        self.pos_entries_dic = {}
        for i in range(no):
            self.split(X, Y)
        split_arr = [self.splits[0][2]]
        if (no > 1):
            for i in range(1, len(self.splits)):
                split_arr = np.append(split_arr, [self.splits[i][2]], axis=0)
        split_arr = split_arr.transpose()
        iter_all = [comb for iters in list(itertools.product([True, False], repeat=i + 1) for i in range(no)) for comb in iters]
        pos_entries = [s for s in iter_all if len(s) == no]
        ind_list = []
        for i in range(len(pos_entries)):
            tmp_list = []
            for l in range (len(X)):
                tmp_list.append(np.all(split_arr[l] == pos_entries[i]))
            ind_list.append(tmp_list)
        Y_list = []
        for i in range(len(ind_list)):
            Y_list.append(Y[ind_list[i]])
        prob_list = []
        whole_prob = sum(Y[:]==1)/len(X)
        print("Whole prob: ", whole_prob)
        for i in range(len(ind_list)):
            num = sum(Y_list[i] == 1)
            denom =len(Y_list[i])
            if denom != 0.0:
                prob_list.append(num/denom)
            else:
                prob_list.append(whole_prob)
        self.pos_entries_dic = dict.fromkeys(pos_entries, 0)
        for iter in self.pos_entries_dic:
            self.pos_entries_dic[iter] = prob_list.pop(0)

    def split(self, X, Y):
        # split under random axis
        split_axis = np.random.randint(0, X.shape[1]) # random split axis of all possible axis
        unique_val = np.unique(X[:, split_axis]) # random value of all possible split value
        split_val = unique_val[np.random.randint(0, len(unique_val))]
        px, nx, = self.split_set(X, Y, split_axis, split_val)
        self.splits.append([split_axis, split_val, px, nx])

    def split_set(self, X, Y, axis, val):
        px = X[:, axis] == val
        nx = X[:, axis] != val

        pos_Y = Y[px]
        neg_Y = Y[nx]
        # calculate the probabilities for each side
        return px, nx
    
    def pred(self, X, Y):
        px_list = []
        for split in self.splits:
            px, nx = self.split_set(X, Y, split[0], split[1])
            px_list.append(px)
        split_arr = [px_list[0]]
        if (len(px_list) > 1):
            for i in range(1, len(px_list)):
                split_arr = np.append(split_arr, [px_list[i]], axis=0)
        split_arr = split_arr.transpose()
        pred_true = 0
        for i in range(len(X)):
            if self.pos_entries_dic[tuple(split_arr[i])] > 0.5 and Y[i] == 1:
                pred_true += 1
            elif self.pos_entries_dic[tuple(split_arr[i])] < 0.5 and Y[i] == -1:
                pred_true += 1
        print(pred_true, " right predictions from ", len(X), " , acc= ", pred_true/len(X))



def read_data(f_name='mushrooms.csv'):
    file = open(f_name)
    reader = csv.reader(file)
    data = []
    for row in reader:
        data.append(row)
    data = np.array(data)
    Y = data[1:, 0]
    Y = ((Y == 'e') + 0.) * 2 - 1
    X = data[1:, 1:]
    feature_names = data[0, :]
    file.close()
    return X, Y, feature_names

# create training/testing/evaluation sets
X, Y, features_names = read_data();
ratio_train = .6
ratio_eval = .2
ratio_test = 1-(ratio_train+ratio_eval)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

fern = Fern(X_train, y_train, 7)
# fern.pred(X_test, y_test)



