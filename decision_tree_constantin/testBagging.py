# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 12:28:16 2017

@author: Henning
"""
from __future__ import division
from ensembleMethods import BaggedLearner
from ensembleMethods import AdaBoost
from continuousDTLearner import Learner
from matplotlib import pyplot as plt
from scipy.misc import imread
import csv
import numpy as np
import sklearn

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
def one_hot(X,Y):
    newNrFeat = 0
    for i in range(X.shape[1]):
        feat = np.unique(X[:,i])
        newNrFeat += len(feat)
    newX = np.empty((X.shape[0],newNrFeat))
    index = 0
    for i in range(X.shape[1]):
        feat = np.unique(X[:,i])
        for j,f in enumerate(feat):
            newFeatures = X[:,i] == f * 1
            newX[:,index] = newFeatures
            index += 1
            newCSV = np.hstack((Y.reshape(Y.shape[0],1),newX))
    np.savetxt("mushroomsOneHot.csv",newCSV,delimiter = ",")    


#X, Y, feature_names,code = read_data(f_name = "mushroomsOneHot.csv")
#X, Y, feature_names,code = read_data(f_name = "wdbc.csv")
#print(X.shape, Y.shape)
X, Y = read_image()
#best_features = [4,11,12,8]
#worst_features = [x for x in np.arange(X.shape[1]) if not (x==best_features).any()]
#X = X[:,worst_features]

X_train,Y_train,_,_,_,_ = split_sets(X,Y,ratio_train=1,ratio_test=0)
_,_,X_test,Y_test,_,_ = split_sets(X,Y,ratio_train=0,ratio_test=1)
#X_train,Y_train,X_test,Y_test,_,_ = split_sets(X,Y,ratio_train=0.7,ratio_test=0.3)
print("generated Data")
maxEnsSize = 4
perfRes = np.zeros(maxEnsSize)
for j,ensSize in enumerate(np.arange(maxEnsSize)+1):
    avgPerf = 0
    nrIt = 1
    for i in range(nrIt): 
#        X, Y, feature_names,code = read_data(f_name = "wdbc.csv")
#        best_features = [4,11,12,8]
#        worst_features = [x for x in np.arange(X.shape[1]) if not (x==best_features).any()]
#        X = X[:,worst_features]
#
#        X_train,Y_train,X_test,Y_test,X_eval,Y_eval = split_sets(X,Y,ratio_train=.4,ratio_test=.2)
   
#sklearn.ensemble.AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)[source]

    
        #learner = BaggedLearner(X_train, Y_train,"ddddddddddddddddddddddddddddddddddddddddddd",feature_names,max_depth = 2,ensSize=10)
        learner = AdaBoost(X_train, Y_train,ensSize=ensSize)
        #learner = Learner(X_train, Y_train,"dddddddddddddddddddddddddddddddddddddddddddddd",feature_names,max_depth = 1)
        print("finnished learning")
        
        errors = 0
        for i,xt in enumerate(X_test):
#            if learner.predict(xt) != -1:
#                print("nicht 1")
            if not learner.predict(xt) == Y_test[i]:
                #print(learner.predict(xt),Y_test[i])
                errors+=1
        perf = 1 - errors/X_test.shape[0]
        #print("Performance: ",  perf)
        avgPerf += perf/nrIt
    print("AvgPerformance: ",  avgPerf)
    perfRes[j] = avgPerf
fig = plt.figure("Results AdaBoost")  
#ax = fig.add_axes()
plt.plot(np.arange(maxEnsSize)+1,perfRes,"bo")
plt.xlabel("Ensemble Size", fontsize=18)
plt.ylabel("Performance", fontsize=18)
fig.savefig('Results AdaBoost '+str(maxEnsSize)+" "+str(nrIt)+'.png')
#ax.tick_params(labelsize=10)
plt.show()
        


    
    
    
    