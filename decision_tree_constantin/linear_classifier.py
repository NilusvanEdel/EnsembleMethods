import numpy as np


class LinearClassifier():

	def __init__(self,X,Y):
		self.X = X
		self.Y = Y
		self.train()

	def train(self):
		x_inv = np.linalg.pinv(self.X)
		self.weights = np.dot(x_inv,self.Y)


	def predict(self,X):
		Y = []
		for x in X:
			Y.append(np.sign(np.dot(self.weights.T,x)))
		return np.squeeze(Y)
			