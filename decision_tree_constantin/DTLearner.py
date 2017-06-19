import numpy as np
from math import log as mlog

class Batch():
	'''
	Deals samples in batches
	'''
	def __init__(self,L):
		X = L[0]
		Y = L[1]
		perm = np.random.permutation(X.shape[0])
		self.X = X[perm,:]
		self.Y = Y[perm]

	def next(self, size):
		idx = np.random.permutation(self.X.shape[0])[:size]
		x = self.X[idx,:]
		y = self.Y[idx]

		#self.X = self.X[size:,:]
		#self.Y = self.Y[size:]

		return x,y 

class DecisionNode:
	'''
	implements a decision node, i.e. a node that has (two) children.
	The positive child is visited, iff the datapoint has a certain
	value at a certain axis.
	'''
	def __init__(self, axis = None, val = None):
		self.pos_child = None
		self.neg_child = None
		self.axis = axis
		self.val = val
		self.type = 'Decision Node'

class LeafNode:
	'''
	Leaf nodes have no children, but a class label
	'''
	def __init__(self, label, axis = None, val = None):
		self.axis = axis
		self.val = val
		self.label = label
		self.type = 'Leaf Node'


class DecisionTree:
	'''
	Baiscally only the root node as decision node
	max_depth is not implemented right now
	'''
	def __init__(self):
		self.root = DecisionNode()



class Learner:
	'''
	The actual decision tree learner.
	'''

	def __init__(self, max_depth = 3):
		self.tree = DecisionTree()
		self.max_depth = max_depth

	def opposite(self,x):
		return -x

	def most_frequent_label(self,Y):
		ys = np.unique(Y)
		count = []
		for y in ys:
			count.append(np.sum(Y==y))
		mfl = ys[np.argmax(count)]
		return mfl

	def entropy(self,Y):
		if all(Y==Y[0]):
			return 0
		labels = np.unique(Y)
		label_probs = []
		for l in labels:
			label_probs.append(np.mean(Y==l))
		entropy = 0
		for p in label_probs:
			entropy -= p * mlog(p, len(labels))
		return entropy


	def information_gain(self,X,Y):
		# first, calculate global entropy
		global_entropy = self.entropy(Y)
		# for each possible given class label, calculate the entropy
		# multiply by chance to get label
		labels = np.unique(X)
		information = 0
		label_gain = []
		for label in labels:
			p_label = np.mean(X==label)
			per_label = self.entropy(Y[X==label]) * p_label
			information += per_label
			label_gain.append(per_label)
		gain = global_entropy - information
		best_label = labels[np.argmax(label_gain)] 
		return gain, best_label

	def get_split_gains(self,X,Y):
		'''
		Information gain of a split is calculated based on the ratio of 
		positive and negative examples before and after applying the split.
		'''
		
		# for each axis x in X and each feature value in x calculate the
		# information gain
		max_gain_per_axis = []
		best_feature_vals = []
		for axis in range(X.shape[1]):
			x = X[:,axis]
			inf_gain, label = self.information_gain(x,Y)
			max_gain_per_axis.append(inf_gain)
			best_feature_vals.append(label)
		return max_gain_per_axis, best_feature_vals

	def get_split(self,X,Y):
		split_gains, split_val = self.get_split_gains(X,Y)
		axis = np.argmax(split_gains)
		val = split_val[axis]
		return axis, val

	def split_set(self,X,Y,axis,val):
		ix = X[:,axis]==val
		jx = X[:,axis]!=val

		pos_X = X[ix]
		neg_X = X[jx]

		pos_Y = Y[ix]
		neg_Y = Y[jx]

		return pos_X, pos_Y, neg_X, neg_Y

	def build_tree(self,X,Y,cur_node,depth):
		# find good split axis
		axis, val = self.get_split(X,Y)
		# give parameters to node
		cur_node.axis = axis
		cur_node.val = val
		# apply split
		pos_X, pos_Y, neg_X, neg_Y = self.split_set(X,Y,axis,val)

		if depth >= self.max_depth-1:
			cur_node.pos_child = LeafNode(self.most_frequent_label(pos_Y))
			cur_node.neg_child = LeafNode(self.opposite(cur_node.pos_child.label))
			return

		if pos_Y.shape[0]==0:
			cur_node.neg_child = LeafNode(self.most_frequent_label(neg_Y))
			cur_node.pos_child = LeafNode(self.opposite(cur_node.neg_child.label))
			return
		if neg_Y.shape[0]==0:
			cur_node.pos_child = LeafNode(self.most_frequent_label(pos_Y))
			cur_node.neg_child = LeafNode(self.opposite(cur_node.pos_child.label))
			return
		if all(pos_Y == pos_Y[0]):
			cur_node.pos_child = LeafNode(pos_Y[0])
		else:
			cur_node.pos_child = DecisionNode()
			self.build_tree(pos_X,pos_Y,cur_node.pos_child,depth+1)
		if all(neg_Y == neg_Y[0]):
			cur_node.neg_child = LeafNode(neg_Y[0])
		else:
			cur_node.neg_child = DecisionNode()
			self.build_tree(neg_X,neg_Y,cur_node.neg_child,depth+1)


	def init_tree(self,X,Y):
		self.build_tree(X,Y,self.tree.root,0)



	def get_node_info(self,node):
		return node.axis, node.val, node.label


	def recursive_print(self,node,mode,indent):
		indent += 1
		lead = '	'*indent

		if not type(node)==LeafNode:
			print(lead+'+-----------'.format(indent))
			print(lead+'| '+ node.type)
			print(lead+'| level: {0}'.format(indent))
			print(lead+'| type:  '+mode)
			print(lead+'| axis:  '+str(node.axis))
			print(lead+'| val :  '+str(node.val))
			print(lead+'+-----------'.format(indent))
			if node.pos_child is None:
				print('  '*indent+'|no pos child')
			else:
				self.recursive_print(node.pos_child,'pos',indent)
		else:
			print(lead+'+-----------'.format(indent))
			print(lead+'| type: '+node.type)
			print(lead+'| level: {0}'.format(indent))
			print(lead+'| type:  '+mode)
			print(lead+'| label: '+str(node.label))
			print(lead+'+-----------'.format(indent))


		if not type(node)==LeafNode:
			if node.neg_child is None:
				print('  '*indent+'|no neg child')
			else:
				self.recursive_print(node.neg_child,'neg',indent)


	def print_tree(self):
		cur_node = self.tree.root
		self.recursive_print(cur_node,'root',-1)

	def predict(self,x):
		n = self.tree.root

		depth = 0
		while not type(n) == LeafNode:
			self.cur_node = n
			if x[n.axis]==n.val:
				n = n.pos_child
			else:
				n = n.neg_child
			if type(n)==LeafNode:
				return n.label
			depth+=1

		return n.label
