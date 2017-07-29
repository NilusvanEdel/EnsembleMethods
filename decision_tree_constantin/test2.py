import numpy as np 
import csv
import continuousDTLearner as cDTL
import copy
import perceptron as p
import linear_classifier as lc
from math import log as mlog
import matplotlib.pyplot as plt
from scipy.misc import imread
from batch import Batch



def read_data(f_name = 'min_tit3.csv'):
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
    file.close()
    return X, Y, feature_names
 

def read_image(f_name = 'test.png'):
    data = imread(f_name)
    if len(data.shape)>2:
        data = data[:,:,0]
    data = np.array([(data[i,j],i,j) for i in range(data.shape[0]) for j in range(data.shape[1])])
    X = data[:,1:]
    Y = data[:,0]
    vals = np.unique(Y)
    code = np.arange(0,len(vals))
    for ix,v in enumerate(vals):
        Y[Y==v] = code[ix]
    Y = Y*2-1
    return X,Y

def split_sets(X,Y,ratio_train=.4,ratio_test=.2,randomize = True):
    if randomize:
        idx = np.random.permutation(Y.shape[0])
        Y = Y[idx]
        X = X[idx]
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


X,Y, = read_image()

ratio_train = .99
ratio_test = .01

X_train,Y_train,X_test,Y_test,X_eval,Y_eval = split_sets(X,Y,
                                ratio_train=ratio_train,ratio_test=ratio_test)

tree = cDTL.Learner(X_train,Y_train,['c','c'], max_depth = 20)

K = np.reshape(Y,[100,100])
plt.imshow(K,extent=[-1,99,99,-1])
tree.print2d()