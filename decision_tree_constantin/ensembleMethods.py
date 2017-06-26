# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:56:36 2017

@author: Henning
"""
from continuousDTLearner import Learner
import numpy as np

class BaggedLearner:
    
    def __init__(self,X,Y,feature_types,feature_names=None,max_depth = 3,ensSize=10):
        self.ensSize = ensSize
        self.X = X
        self.Y = Y
        self.learners = []
        for i in range(ensSize):
            bagInd = self.drawBag()
            self.learners.append(Learner(X[bagInd],Y[bagInd],feature_types,feature_names,max_depth = 3))

    """
    draw indices of the surrogate dataset
    """
    def drawBag(self):
        bagInd = np.random.random_integers(low = 0, high = self.X.shape[0]-1,size=self.X.shape[0])
        return bagInd
    
    def learn(self):
        for learner in self.learners:
            learner.init_tree(self.X,self.Y)
            
    def predict(self,x):
        predictions = []
        for learner in self.learners:
            predictions.append(learner.predict(x))
        return max(set(predictions), key=predictions.count)
            
        