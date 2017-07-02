# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 12:28:16 2017

@author: Henning
"""
from __future__ import division
from ensembleMethods import BaggedLearner
from ensembleMethods import AdaBoost
from continuousDTLearner import Learner
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
    return X[:,10::], Y, feature_names,code

def split_sets(X,Y,ratio_train=.4,ratio_test=.2):
	l = Y.shape[0]
	X_train = X[:int(l*ratio_train),:]
	X = X[int(l*ratio_train):,:]
	Y_train = Y[:int(l*ratio_train)]
	Y = Y[int(l*ratio_train):]
	X_test = X[:int(l*ratio_test),:]
	X = X[int(l*ratio_test):,:]
	Y_test = Y[:int(l*ratio_test)]
	Y = Y[int(l*ratio_test):]
	X_eval = X
	Y_eval = Y

	return X_train,Y_train,X_test,Y_test,X_eval,Y_eval

X, Y, feature_names,code = read_data(f_name = "mushrooms.csv")
#X, Y, feature_names,code = read_data(f_name = "mushrooms.csv")
X_train,Y_train,X_test,Y_test,X_eval,Y_eval = split_sets(X,Y,ratio_train=.4,ratio_test=.2)
#learner = BaggedLearner(X_train, Y_train,"ddddddddddddddddddddddddddddddddddddddddddd",feature_names,max_depth = 2,ensSize=10)
learner = AdaBoost(X_train, Y_train,"dddddddddddddddddddddddddddddddddddddddddd",feature_names,max_depth = 1,ensSize=10)
#learner = Learner(X_train, Y_train,"dddddddddddddddddddddddddddddddddddddddddddddd",feature_names,max_depth = 2)


errors = 0
for i,xt in enumerate(X_test):
    if not learner.predict(xt) == Y_test[i]:
        errors+=1
print("Performance: ", 1 - errors/X_test.shape[0])
    
    
    
    