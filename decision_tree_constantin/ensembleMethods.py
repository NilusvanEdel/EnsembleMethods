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
            #print("in sample error: ",np.mean(losses))
            #compute the weighted error
            wError =  self.calcWeightedError(losses*1,beta)
            #print("wError: ",wError)
            #compute hypothesis weight
            self.calcHypoWeight(wError,i)
            #update the weights beta
            beta = self.updateBeta(losses,wError,i,beta)
            #self.plot(beta,i)
        print("hypoWeights: ", self.hypoWeights)
        
        
    def learn(self,i,beta):

        l = stumpLearner(self.X,self.Y,beta)
        l.learn()
        self.learners[i] = l
                     
    def predictions(self,learner):
        predictions = np.empty(self.N)
        for i,x in enumerate(self.X):
            predictions[i] = learner.predict(x)
        return predictions
        
    def calcLosses(self,learner):
        losses = np.empty(self.N,dtype=bool)
        for i,x in enumerate(self.X):
            losses[i] = learner.predict(x) != self.Y[i]
        return losses
    
    def calcWeightedError(self,losses,beta):
        return np.dot(beta, losses)
    
    def calcHypoWeight(self,wError,i):
        #SE solution:
#        eps = self.beta.dot(self.predictions(self.learners[i]) != self.Y)
#        self.hypoWeights[i] = (np.log(1 - eps) - np.log(eps)) / 2

        if np.allclose(wError, 1):
            raise RuntimeError("wError is 1")
            #print("in 1")
            self.hypoWeights[i] = 0
        elif np.allclose( wError, 0):
            raise RuntimeError("wError is 0")
            #TODO: was tun wenn error 0?
            self.hypoWeights[i] = 10000
        else:
            #print("in 3")
            self.hypoWeights[i] = 0.5 * np.math.log((1-wError) / wError)
        
    def updateBeta(self,losses,wError,i,beta):
        #beta = beta * np.exp(- self.hypoWeights[i] * self.Y * self.predictions(self.learners[i]))
        #beta *= np.exp((losses*1)*wError)
        #for worngly classified beta:
        beta[losses] *= np.exp(self.hypoWeights[i])
        #self.beta[losses] /= 2*wError
                 
        #for correctly classified beta:
        beta[np.logical_not(losses)] *= np.exp(-self.hypoWeights[i])
        #self.beta[np.logical_not(losses)] /= 2*(1-wError)
        #normalize
        beta /= np.sum(beta)
        if (beta < 0).any():
            raise RuntimeError("beta kleiner 0")
        return beta

    def predict(self,x):
        predictions = np.empty_like(self.learners)
        for i,learner in enumerate(self.learners):
            predictions[i] = learner.predict(x)
#        if not (predictions == predictions[0]).all():
#            print("single predictions ",predictions)
        pred = self.voting(predictions)
        #print("pred out: ", pred)
        return pred
    
#    def voting(self,predictions):
#        preSet = set(predictions)
#        lWeights = np.empty(len(preSet),dtype=np.float64)
#        for i,l in enumerate(preSet):
#            lWeight = np.sum(self.hypoWeights[predictions == l])
#            lWeights[i] = lWeight
#        return list(preSet)[np.argmax(lWeights)]
    def voting(self,predictions):
        return np.sign(np.dot(self.hypoWeights,predictions))
    def plot(self,beta,i):
        plt.figure("fig "+str(i))
        shape = int(np.sqrt(self.Y.shape[0]))
        Yplot = self.Y * beta
        plt.imshow(Yplot.reshape((shape,shape)))
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
            
        