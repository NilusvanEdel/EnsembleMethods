import numpy as np

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