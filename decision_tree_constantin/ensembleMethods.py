# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:56:36 2017

@author: Henning
"""
from __future__ import division
from stumpLearner import stumpLearner
import numpy as np
import threading
from matplotlib import pyplot as plt
"""
AdaBoost Learner
"""
class AdaBoost:
    
    def __init__(self,X,Y,ensSize=10):
        self.X = X
        self.Y = Y
        self.learners = np.empty(ensSize,dtype=object)
        self.hypoWeights = np.empty(ensSize,dtype=np.float16)
        self.N = self.X.shape[0]

        beta = np.ones(self.N) * (1./self.N)
        for i in range(ensSize):
            self.learn(i,beta)
            #calculate losses
            losses = self.calcLosses(self.learners[i])
            #compute the weighted error
            wError =  self.calcWeightedError(losses*1,beta)
            #compute hypothesis weight
            self.calcHypoWeight(wError,i)
            #update the weights beta
            beta = self.updateBeta(losses,wError,i,beta)
            #self.plot(beta,i)
        
    """
    start learning
    """
    def learn(self,i,beta):

        l = stumpLearner(self.X,self.Y,beta)
        l.learn()
        self.learners[i] = l

    def calcLosses(self,learner):
        losses = np.empty(self.N,dtype=bool)
        for i,x in enumerate(self.X):
            losses[i] = learner.predict(x) != self.Y[i]
        return losses

    def calcWeightedError(self,losses,beta):
        return np.dot(beta, losses)
    
    def calcHypoWeight(self,wError,i):
        self.hypoWeights[i] = 0.5 * np.math.log((1-wError) / wError)
        
    def updateBeta(self,losses,wError,i,beta):
        #for worngly classified beta:
        beta[losses] *= np.exp(self.hypoWeights[i])
                 
        #for correctly classified beta:
        beta[np.logical_not(losses)] *= np.exp(-self.hypoWeights[i])
        
        #normalize
        beta /= np.sum(beta)
        return beta

    def predict(self,x):
        predictions = np.empty_like(self.learners)
        for i,learner in enumerate(self.learners):
            predictions[i] = learner.predict(x)
        pred = self.voting(predictions)
        return pred
    
    def voting(self,predictions):
        return np.sign(np.dot(self.hypoWeights,predictions))
    
    def plot(self,beta,i):
        plt.figure("fig "+str(i))
        shape = int(np.sqrt(self.Y.shape[0]))
        Yplot = self.Y * beta
        plt.imshow(Yplot.reshape((shape,shape)))
        #plot split axes
        for j,l in enumerate(self.learners[0:i+1]):
            split = l.split
            if split[0] == 0:
                plt.axhline(split[1],linewidth = 3,color = (0.3*j,0.3*j,0.3*j),label = "split "+str(j+1))
            else:
                plt.axvline(split[1],linewidth = 3,color = (0.3*j,0.3*j,0.3*j),label = "split "+str(j+1))
        plt.legend()
        if i == len(self.learners)-1:
            plt.savefig("adaBoost2Dsplits.png")
class BaggedLearner:
    
    def __init__(self,X,Y,feature_types,feature_names=None,max_depth = 3,ensSize=10,random_splits = False):
        self.X = X
        self.Y = Y
        self.learners = []
        self.feature_types = feature_types
        self.feature_names = feature_names
        self.max_depth = max_depth
        self.random_splits = random_splits
        self.seed = 0
        threads = []
        for i in range(ensSize):
            t = threading.Thread(target=self.learn,args =(i,))
            self.seed += 1
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
            
    def learn(self,i):
        seed = np.random.randint(0,1000000)
        np.random.seed(seed)
        bagInd = self.drawBag()
        #get random features
        if self.random_splits:
            feat = np.arange(self.X[bagInd].shape[1])
            feat = np.random.choice(feat,size=int(len(feat)/5 + 1),replace=False)
            #l = cDTL.Learner(max_depth = self.max_depth, random_splits = False, feature_names=self.feature_names)
            #l.learn(self.X[bagInd][:,feat],self.Y[bagInd],self.feature_types)
            l = stumpLearner(self.X[bagInd][:,feat],self.Y[bagInd],beta= 1 / bagInd.shape[0]*np.ones(bagInd.shape[0]))
            l.learn()
        else:
            #l = cDTL.Learner(max_depth = self.max_depth, random_splits = False, feature_names=self.feature_names)
            #l.learn(self.X[bagInd],self.Y[bagInd],self.feature_types)
            l = stumpLearner(self.X[bagInd],self.Y[bagInd],beta= 1 / bagInd.shape[0]*np.ones(bagInd.shape[0]))
            l.learn()
        self.learners.append(l)

    """
    draw indices of the surrogate dataset
    """
    def drawBag(self):
        bagInd = np.random.random_integers(low = 0, high = self.X.shape[0]-1,size=self.X.shape[0])
        return bagInd
    
            
    def predict(self,x):
        predictions = np.empty(len(self.learners))
        weights = np.empty(len(self.learners))
        for i,learner in enumerate(self.learners):
            predictions[i] = learner.predict(x)
            weights[i] = learner.split[2]
        return np.sign(np.dot(predictions,weights))
        
        