import numpy as np
import matplotlib.pyplot as plt
import cDTLearner as cDTL
from dataInformation import DataInformation
from dataHelper import Batch, read_data, split_sets


np.random.seed(1)

# read in data
_,_,feature_names,codeing = read_data(f_name = 'wdbc.csv')

#X[:,[4,11,12,8,7,18,19,3,20,6,21,14]] = '0'

feature_names = feature_names[1:]
# split into training, testing, and validation sets
#X_train,Y_train,X_test,Y_test,X_val,Y_val = split_sets(X,Y,ratio_train=.7,ratio_test=.25)
di = DataInformation(True)
X_train, X_test, X_val, Y_train, Y_test, Y_val = di.get_TestTrainVal(0.3, 0.33, False)

tmp = Y_train[0]
Y_train = (Y_train == tmp)*2-1
Y_test = (Y_test == tmp)*2-1
Y_val = (Y_val == tmp)*2-1


# specify the type of features
feature_types = 'd'*X_train.shape[1]
#feature_types = 'dddddcc'


# initialize a tree
tree_regular = cDTL.Learner(max_depth = 5,
					random_splits = True,
					feature_indices = None,
					feature_names = feature_names)
# train the tree
tree_regular.learn(np.copy(X_train),np.copy(Y_train),feature_types)
# test the tree
Y_hat = [tree_regular.predict(x) for x in X_test]
perf = np.mean(Y_hat==Y_test)
print('performance regular tree: {0}'.format(perf))
jdsai
'''
Here we initialize a tree with predifined features to split along
'''
feature_subset = np.random.permutation(X_train.shape[1])[:3]
# initialize a tree
tree_pred_feats = cDTL.Learner(max_depth = 5,
					random_splits = False,
					feature_indices = feature_subset,
					feature_names = feature_names)

# train the tree
tree_pred_feats.learn(np.copy(X_train),np.copy(Y_train),feature_types)



# test the tree
Y_hat = [tree_pred_feats.predict(x) for x in X_test]
perf = np.mean(Y_hat==Y_test)
print('performance tree using only features {0}: {1}'.format(feature_subset,perf))


'''
Here we initialize a tree with random splits along the same feature axes as before
'''
# initialize a tree
tree_rand = cDTL.Learner(max_depth = 5,
					random_splits = True,
					feature_indices = feature_subset,
					feature_names = feature_names)

# train the tree
tree_rand.learn(np.copy(X_train),np.copy(Y_train),feature_types)


# test the tree
Y_hat = [tree_rand.predict(x) for x in X_test]
perf = np.mean(Y_hat==Y_test)
print('performance tree using only features {0} and random splits: {1}'.format(feature_subset,perf))


'''
Random Forest Time
'''

gain,_ = tree_regular.get_split_gains(X_train,Y_train)
K = np.flipud(np.argsort(gain))

ps1 = []
ps2 = []
ps3 = []
for k in range(len(K)):
	print(K[:k])
	n_trees = 500
	m_depth = 1
	fpt = None
	xf = np.vstack((X_train,X_test))
	#[4,11,12,8,7,18,19,3,20,6,21,14]
	xf[:,K[:k]] = 0
	yf = np.hstack((Y_train,Y_test))
	forest = cDTL.ForestLearner(xf,yf,
								feature_types, 
								max_depth = m_depth,
								n_trees = n_trees,
								share_features = True, 
								ratio_train = 0.7,
								random_splits = True,
								batch_size = 50,
								features_per_tree = fpt,
								feature_names = feature_names)
	Y_hat = np.array([forest.predict(x) for x in X_val])
	Y_hat_linear = np.array([forest.predict_linear(x) for x in X_val])
	Y_hat_vote = np.array([forest.predict_vote(x) for x in X_val])
	perf = np.mean(Y_hat==Y_val)
	ps1.append(perf)
	print('performance random forest ({0} trees with depth {1}) and tree as ensemble combination: {2}'.format(n_trees,m_depth,perf))
	perf = np.mean(Y_hat_linear==Y_val)
	ps2.append(perf)
	print('performance random forest ({0} trees with depth {1}) and only linear combination: {2}'.format(n_trees,m_depth,perf))
	perf = np.mean(Y_hat_vote==Y_val)
	ps3.append(perf)
	print('performance random forest ({0} trees with depth {1}) and by vote under trees: {2}'.format(n_trees,m_depth,perf))




'''
test effect of dropout of features
'''

# order features by information gain

x = np.copy(X_train)
y = np.copy(Y_train)
xtmpval = np.copy(X_val)

perfs = []
dropded_features = []
ftmp = feature_names
for t in range(X_train.shape[1]):
	gains, _ = tree_regular.get_split_gains(x,y)
	idx = np.flipud(np.argsort(gains))
	tree_dropout = cDTL.Learner(max_depth = 5)
	tree_dropout.learn(x,y,feature_types)
	Y_hat = [tree_dropout.predict(x) for x in xtmpval]
	perf = np.mean(Y_hat==Y_val)
	x = x[:,idx[1:]]
	xtmpval = xtmpval[:,idx[1:]]
	dropded_features.append(ftmp[idx[0]])
	ftmp = ftmp[idx[1:]]
	perfs.append(perf)

plt.plot(perfs)
plt.xticks(np.arange(1,len(feature_names)+1,1),dropded_features,rotation='vertical')
plt.title('ordered feature dropout')
plt.show()



'''

this part simulates a fern with 'n_splits' splits

'''
"""
ps = []

for it in range(4):
	pp = []
	for n_splits in np.arange(1,16,1):
		forest = cDTL.ForestLearner(X_train,Y_train,
							feature_types, 
							max_depth = 1,
							n_trees = n_splits,
							share_features = True, 
							ratio_train = 0.7,
							random_splits = True,
							features_per_tree = None,
							feature_names = feature_names)
		Y_hat = np.array([forest.predict(x) for x in X_test])
		p = np.mean(Y_hat==Y_test)
		pp.append(p)
	ps.append(pp)

ps = np.mean(np.array(ps),0)


plt.bar(np.arange(1,ps.shape[0]+1,1),ps)
plt.show()
"""


