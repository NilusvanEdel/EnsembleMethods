# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:56:36 2017

@author: Henning
"""
from continuousDTLearner import Learner
import numpy as np
import threading

class AdaBoost:
    
    def __init__(self,X,Y,feature_types,feature_names=None,max_depth = 3,ensSize=10):
        self.X = X
        self.Y = Y
        self.learners = []
        self.hypoWeights = []
        self.feature_types = feature_types
        self.feature_names = feature_names
        self.max_depth = max_depth
        self.N = self.X.shape[0]

        self.beta = np.ones(self.N) * (1./self.N)
        for i in range(ensSize):
            self.learn()
            #calculate losses
            losses = self.loss(self.learners[-1])
            #compute the weighted error
            wError =  self.calcWeightedError(losses*1)
            #compute hypothesis weight
            self.calcHypoWeights(wError)
            #update the weights beta
            self.updateBeta(losses)
        
        
    def learn(self):

        l = Learner(self.X,self.Y,self.feature_types,self.feature_names,self.max_depth,self.beta)
        self.learners.append(l)
        
    def loss(self,learner):
        predictions = np.empty((self.N,0),dtype=int)
        for i,x in enumerate(self.X):
            predictions[i] = learner.predict(x)
        return np.logical_not(predictions == self.Y)
    
    def calcWeightedError(self,losses):
        return np.sum(self.beta * losses)
    
    def calcHypoWeights(self,wError):
#        wError = np.float64(wError) + 0
#        print(np.dtype(wError))
#        print("wError ",np.float64(wError))
#        print(np.float64(wError) is np.float64(1.0))

        if np.allclose(wError, 1):
            #print("in 1")
            self.hypoWeights.append(0)
        elif np.allclose( wError, 0):
            #TODO: was tun wenn error 0?
            self.hypoWeights.append(10000)
        else:
            #print("in 3")
            self.hypoWeights.append(0.5 * np.math.log((1-wError) / wError))
        
    def updateBeta(self,losses):
        #for worngly classified beta:
        self.beta[losses] *= np.exp(self.hypoWeights[-1])
        #for correctly classified beta:
        self.beta[np.logical_not(losses)] *= np.exp(-self.hypoWeights[-1])
        #normalize
        self.beta /= np.sum(self.beta)

    def predict(self,x):
        predictions = []
        for learner in self.learners:
            predictions.append(learner.predict(x))
        return self.voting(predictions)
    
    def voting(self,predictions):
        return max(set(predictions), key=predictions.count)

class BaggedLearner:
    
    def __init__(self,X,Y,feature_types,feature_names=None,max_depth = 3,ensSize=10):
        self.X = X
        self.Y = Y
        self.learners = []
        self.feature_types = feature_types
        self.feature_names = feature_names
        self.max_depth = max_depth
        threads = []
        for i in range(ensSize):
            t = threading.Thread(target=self.learn)
            t.start()
            threads.append(t)
            print("started thread: ",i+1)
        for t in threads:
            t.join()
            print("done")
            
    def learn(self):

        bagInd = self.drawBag()
        l = Learner(self.X[bagInd],self.Y[bagInd],self.feature_types,self.feature_names,self.max_depth)
        self.learners.append(l)

    """
    draw indices of the surrogate dataset
    """
    def drawBag(self):
        bagInd = np.random.random_integers(low = 0, high = self.X.shape[0]-1,size=self.X.shape[0])
        return bagInd
    
#    def learn(self):
#        for i,learner in enumerate(self.learners):
#            threading.Thread(target=learner.init_tree,args = (self.X,self.Y)).start()
#            print("started thread: ",i)
            
    def predict(self,x):
        predictions = []
        for learner in self.learners:
            predictions.append(learner.predict(x))
        return self.voting(predictions)
    
    def voting(self,predictions):
        return max(set(predictions), key=predictions.count)
            
        