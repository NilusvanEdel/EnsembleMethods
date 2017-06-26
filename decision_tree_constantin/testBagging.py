# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 12:28:16 2017

@author: Henning
"""

from ensembleMethods import BaggedLearner
import csv
import numpy as np

def read_data(f_name = 'mushrooms.csv'):
    file = open(f_name)
    reader = csv.reader(file)
    data = []
    for row in reader:
        data.append(row)
    data = np.array(data)
    Y = data[1:,0]
    code = np.unique(Y)
    Y = ((Y==code[0])+0.)*2-1
    X = data[1:,1:]
    # clear empty values
    idx = np.invert(np.any(X=='',1))
    X = X[idx,:]
    Y = Y[idx]
    feature_names = data[0,:]
    # shuffle data
    idx = np.random.permutation(Y.shape[0])
    Y = Y[idx]
    X = X[idx,:]
    file.close()
    return X, Y, feature_names,code

X, Y, feature_names,code = read_data()
learner = BaggedLearner(X, Y,"ddddddddddddddddddddddddddddddd",feature_names)
learner.learn()
