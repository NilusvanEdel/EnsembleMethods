import numpy as np
import matplotlib.pyplot as plt
import cDTLearner as cDTL
from dataHelperTitanic import Batch, read_data, split_sets




np.random.seed(10)


# read in data
X,Y,feature_names,codeing = read_data(fill_empties = True)

# specify the type of features
feature_types = 'dddccdcddcdcdc'

# delete entries from X, feature_names, and feature_types
delete = np.array(['length_of_name','parentheses','Surname'])
idx = np.invert([any(delete == f) for f in feature_names])
X = X[:,idx]
feature_names = feature_names[idx]
feature_types = [f for ix,f in enumerate(feature_types) if idx[ix]]



# split into training, testing, and validation sets
X_train,Y_train,X_test,Y_test,X_val,Y_val = split_sets(X,Y,ratio_train=.8,ratio_test=.0)

for d in [1,10,50,100]:
	tree = cDTL.Learner(max_depth = d, random_splits = False, feature_names = feature_names)
	tree.learn(X_train,Y_train,feature_types)
	Y_hat = np.array([tree.predict(x) for x in X_val])
	print('depth {1}: {0}'.format(np.mean(Y_hat==Y_val), d))




t = 100
for d in np.arange(10,110,5):
	# create random forest 
	forest = cDTL.ForestLearner(X_train,Y_train,feature_types,
						forest_depth = int(t/2),
						n_trees = t,
						batch_size = 50,
						max_depth = d,
						ratio_train = .9,
						random_splits = False,
						share_features = True,
						features_per_tree = 5)

	Y_hat = np.array([forest.predict(x) for x in X_val])
	performance = np.mean(Y_hat==Y_val)
	print('performance {2} trees (tree) at depth {1}: {0}'.format(performance,d,t))

	Y_hat = np.array([forest.predict_linear(x) for x in X_val])
	performance = np.mean(Y_hat==Y_val)
	print('performance {2} trees (line) at depth {1}: {0}'.format(performance,d,t))

	Y_hat = np.array([forest.predict_vote(x) for x in X_val])
	performance = np.mean(Y_hat==Y_val)
	print('performance {2} trees (vote) at depth {1}: {0} \n'.format(performance,d,t))