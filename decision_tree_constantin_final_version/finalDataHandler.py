import numpy as np
import csv 


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


def split_sets(X,Y,ratio_train=.4,ratio_test=.2, random_seed = 0):
	np.random.seed(random_seed)
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

def noise(X, level = 0.1):
    np.random.seed(10)
    feature_medians = np.median(X,0)
    noise = []
    for m in feature_medians:
        n = (np.random.uniform(size = X.shape[0])*2-1) * m * level
        noise.append(n)
    Xn = X + np.array(noise).T
    return Xn





def classHistogram(X,Y, n_bins = 50):
    Y = [int(y) for y in Y]
    x = [float(x) for x in X]
    bins = np.linspace(np.min(x),np.max(x)+1e-10, n_bins)
    counts = np.zeros(len(bins))
    vals = [[] for c in counts]
    for ix,p in enumerate(x):
        px = np.max(np.where(p >= bins))
        counts[px] += 1
        vals[px].append(Y[ix])
    vals = vals[:-1]
    vals = np.array([np.mean(v) for v in vals])
    vals[np.isnan(vals)] = 0
    return bins[:-1], counts[:-1], vals