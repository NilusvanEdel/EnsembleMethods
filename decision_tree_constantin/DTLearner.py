import numpy as np

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
		x = self.X[:size,:]
		y = self.Y[:size]

		self.X = self.X[size:,:]
		self.Y = self.Y[size:]

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
		self.type = 'DN'

class LeafNode:
	'''
	Leaf nodes have no children, but a class label
	'''
	def __init__(self, label, axis = None, val = None):
		self.axis = axis
		self.val = val
		self.label = label
		self.type = 'LN'


class DecisionTree:
	'''
	Baiscally only the root node as decision node
	max_depth is not implemented right now
	'''
	def __init__(self, max_depth = 5):
		self.root = DecisionNode()
		self.max_depth = max_depth



class Learner:
	'''
	The actual decision tree learner.
	'''

	def __init__(self):
		self.tree = DecisionTree()
		self.max_depth = 5

	def calc_split_gains(self,X,Y):
		'''
		Information gain of a split is calculated based on the ratio of 
		positive and negative examples before and after applying the split.
		'''
		# calculate entropy of whole dataset
		tmp = np.mean(Y==Y[0])
		# check if data is already only of one class:
		if tmp == 1.0 or tmp == 0.0:
			gloabl_entropy = 0
		else:
			gloabl_entropy = -tmp*np.log2(tmp)-(1-tmp)*np.log2(1-tmp)
		# calculate information gain for every split
		split_gains = []
		split_val = []
		for ix in range(X.shape[1]): # every possible split axis
			variables = np.unique(X[:,ix]) # every possible split value
			gains = []
			for var in variables:
				y = Y[X[:,ix]==var]	# find label for certain value
				tmp = np.mean(y==y[0]) # how many remaining labels are equal
				if tmp == 1.0 or tmp == 0.0:
					entropy = 0
				else:
					entropy = -tmp*np.log2(tmp)-(1-tmp)*np.log2(1-tmp)
				gains.append(gloabl_entropy-entropy) # information gain
			split_gains.append(np.max(gains))
			split_val.append(variables[np.argmax(gains)])
		return split_gains, split_val

	def opposite(self,x):
		return -x

	def most_frequent_label(self,Y):
		ys = np.unique(Y)
		count = []
		for y in ys:
			count.append(np.sum(Y==y))
		mfl = ys[np.argmax(count)]
		return mfl


	def get_split(self,X,Y):
		split_gains, split_val = self.calc_split_gains(X,Y)
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

	def build_tree(self,X,Y,cur_node):
		# find good split axis
		axis, val = self.get_split(X,Y)
		# give parameters to node
		cur_node.axis = axis
		cur_node.val = val
		# apply split
		pos_X, pos_Y, neg_X, neg_Y = self.split_set(X,Y,axis,val)
		# if one set only has one target, create pos/neg child as leaf node 
		if (all(pos_Y==pos_Y[0]) or len(neg_Y) == 0):
			cur_node.pos_child = LeafNode(np.sign(self.most_frequent_label(pos_Y)))
		# else, create child as decision node
		else:
			cur_node.pos_child = DecisionNode()
			self.build_tree(pos_X,pos_Y,cur_node.pos_child)

		if len(neg_Y)==0:
			cur_node.neg_child = LeafNode(self.opposite(pos_Y[0]))
			return

		if (all(neg_Y==neg_Y[0]) or len(pos_Y) == 0):
			cur_node.neg_child = LeafNode(np.sign(self.most_frequent_label(neg_Y)))
		else:
			cur_node.neg_child = DecisionNode()
			self.build_tree(neg_X,neg_Y,cur_node.neg_child)

		# recursive call with pos/neg sets, iff sets not uniform

	def init_tree(self,X,Y):
		self.build_tree(X,Y,self.tree.root)



	def get_node_info(self,node):
		return node.axis, node.val, node.label


	def recursive_print(self,node,mode,indent):
		lead = '  '*indent
		indent += 1

		if not type(node)==LeafNode:
			print(lead+'+-----------'.format(indent))
			print(lead+'| type: '+node.type)
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
		self.recursive_print(cur_node,'root',1)

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
