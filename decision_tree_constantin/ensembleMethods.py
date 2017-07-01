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
        self.feature_types = feature_types
        self.feature_names = feature_names
        self.max_depth = max_depth
        N = self.X.shape[0]
        
        self.beta = np.ones(N) * (1/N)
        for i in range(ensSize):
            self.learn()
            #compute the weighted error
            
            #compute hypothesis weight
            
            #update the weights beta
            
            #normalize weights
        
        
    def learn(self):

        l = Learner(self.X,self.Y,self.feature_types,self.feature_names,self.max_depth,self.beta)
        self.learners.append(l)


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
            
        