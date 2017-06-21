import numpy as np
from batch import Batch
import scipy.io


class Perceptron():
	def __init__(self, X, Y):
		no_weights = X.shape[1]
		self.weights = np.random.uniform(0,size=[no_weights])
		self.tresh = 0
		self.X = X
		self.Y = Y
		self.l_rate = 0.025
		self.train()

	def predict(self,X):
		Y = []
		for x in X:
			Y.append(np.sign(np.dot(self.weights.T,x)))
		return np.squeeze(Y)

	def descent(self,x,y):
		y_hat = np.dot(self.weights.T,x)
		error = y-y_hat
		gradient = (self.l_rate/2) * error * x
		return gradient 

	def train(self,batch_size=50):
		for ix in range(300):
			gradients = []
			for b in range(batch_size):
				rx = np.random.randint(self.X.shape[0])
				x = self.X[rx,:]
				y = self.Y[rx]
				gradient = self.descent(x,y)
				gradients.append(gradient)
			gradients = np.sum(np.array(gradients),0)
			self.weights += (1/batch_size) * np.squeeze(np.array([gradients]))



#
#mat = scipy.io.loadmat('trainingData.mat')
#
#data = mat['U']
#targets = np.squeeze(mat['v'])
#
#
#p = Perceptron(data.shape[1])
#
#p.train(data,targets,1)
#
#r = [np.sign(p.predict(data[ix,:]))==np.sign(targets[ix]) for ix in range(data.shape[0])]
#
#print('correct classified: {0}'.format(np.mean(r)))