import numpy as np 
import csv
import continuousDTLearner as cDTL
import copy
import perceptron as p
import linear_classifier as lc
from math import log as mlog
import matplotlib.pyplot as plt
from scipy.misc import imread



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


def read_image(f_name = 'test4.png'):
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


X,Y, = read_image()
tree = cDTL.Learner(X,Y,['c','c'], max_depth = 100)

K = np.reshape(Y,[100,100])
plt.imshow(K,extent=[-1,99,99,-1])
tree.print2d()