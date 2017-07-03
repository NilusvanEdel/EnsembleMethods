import numpy as np
import matplotlib.pyplot as plt
import cDTLearner as cDTL
from dataHelper import Batch, read_data, split_sets


# read in data
X,Y,feature_names,codeing = read_data(f_name = 'mushrooms.csv')
feature_names = feature_names[1:]
# split into training, testing, and validation sets
X_train,Y_train,X_test,Y_test,X_vali,Y_vali = split_sets(X,Y,ratio_train=.7,ratio_test=.25)

# specify the type of features
feature_types = 'd'*X.shape[1]
#feature_types = 'dddddcc'


# initialize a tree
tree_regular = cDTL.Learner(max_depth = 5,
					random_splits = False,
					feature_indices = None,
					feature_names = feature_names)
# train the tree
tree_regular.learn(X_train,Y_train,feature_types)
# test the tree
Y_hat = [tree_regular.predict(x) for x in X_test]
perf = np.mean(Y_hat==Y_test)
print('performance regular tree: {0}'.format(perf))

gains, _ = tree_regular.get_split_gains(X_train,Y_train)
idx = np.flipud(np.argsort(gains))
print(idx)
print(X_train.shape)


'''
Here we initialize a tree with predifined features to split along
'''
feature_subset = np.random.permutation(X.shape[1])[:3]
# initialize a tree
tree_pred_feats = cDTL.Learner(max_depth = 5,
					random_splits = False,
					feature_indices = feature_subset,
					feature_names = feature_names)

# train the tree
tree_pred_feats.learn(np.copy(X_train),np.copy(Y_train),feature_types)




gains, _ = tree_regular.get_split_gains(np.copy(X_train),np.copy(Y_train))
idx = np.flipud(np.argsort(gains))
print(idx)
print(X_train.shape)


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
n_trees = 30
m_depth = 2
fpt = None
xf = np.vstack((X_train,X_test))
yf = np.hstack((Y_train,Y_test))
forest = cDTL.ForestLearner(xf,yf,
							feature_types, 
							max_depth = m_depth,
							n_trees = n_trees,
							share_features = True, 
							ratio_train = 0.7,
							features_per_tree = fpt,
							feature_names = feature_names)



Y_hat = [forest.predict(x) for x in X_vali]
Y_hat_linear = [forest.predict_linear(x) for x in X_vali]

perf = np.mean(Y_hat==Y_vali)
print('performance random forest ({0} trees with depth {1}): {2}'.format(n_trees,m_depth,perf))

perf = np.mean(Y_hat_linear==Y_vali)
print('performance random forest ({0} trees with depth {1}) and only linear combination: {2}'.format(n_trees,m_depth,perf))


'''
test effect of dropout of features
'''

# order features by information gain

x = np.copy(X_train)
y = np.copy(Y_train)

perfs = []
dropded_features = []
ftmp = feature_names
for t in range(X.shape[1]):
	gains, _ = tree_regular.get_split_gains(x,y)
	idx = np.flipud(np.argsort(gains))
	tree_dropout = cDTL.Learner(max_depth = 5)
	tree_dropout.learn(x,y,feature_types)
	Y_hat = [tree_dropout.predict(x) for x in X_vali]
	perf = np.mean(Y_hat==Y_vali)
	x = x[:,idx[1:]]
	dropded_features.append(ftmp[idx[0]])
	ftmp = ftmp[idx[1:]]
	perfs.append(perf)

plt.plot(perfs)
plt.xticks(np.arange(0,len(feature_names),1),dropded_features,rotation='vertical')
plt.show()