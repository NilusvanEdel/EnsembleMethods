import numpy as np 
import csv
from scipy.misc import imread, imresize


class Batch():
    '''
    Deals samples in batches
    '''
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y

    def next(self, size):
        idx = np.random.permutation(self.X.shape[0])[:size]
        x = self.X[idx,:]
        y = self.Y[idx]

        #self.X = self.X[size:,:]
        #self.Y = self.Y[size:]

        return x,y 

def read_data(f_name = 'wdbc.csv'):
    file = open(f_name)
    reader = csv.reader(file)
    data = []
    for row in reader:
        data.append(row)
    data = np.array(data)
    Y = data[1:,0]
    codeing = np.unique(Y)
    Y = ((Y==codeing[0])+0.)*2-1
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
    return X, Y, feature_names,codeing

def read_image(f_name = 'test.png'):
    data = imread(f_name)
    if len(data.shape)>2:
        data = data[:,:,0]
    data = np.array([(data[i,j],i,j) for i in range(data.shape[0]) for j in range(data.shape[1])])
    X = data[:,1:]
    Y = data[:,0]
    vals = np.unique(Y)
    codeing = np.arange(0,len(vals))
    for ix,v in enumerate(vals):
        Y[Y==v] = codeing[ix]
    Y = Y*2-1
    #idx = np.random.permutation(Y.shape[0])
    #Y=Y[idx]
    #X=X[idx,:]
    return X,Y


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
    X_vali = X
    Y_vali = Y

    return X_train,Y_train,X_test,Y_test,X_vali,Y_vali
