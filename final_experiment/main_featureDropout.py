import numpy as np
import finalDataHandler as dataHandler
import cDTLearner as cDTL 
import matplotlib.pyplot as plt 

# 'wdbc.csv' / 'mushrooms.csv'
data_set_name = 'mushrooms.csv'

# load data
np.random.seed(1)
X, Y, feature_names, codeing = dataHandler.read_data(f_name = data_set_name)

if data_set_name == 'wdbc.csv':
	feature_types = ['c']*X.shape[1]
elif data_set_name == 'mushrooms.csv':
	feature_types = ['d']*X.shape[1]
else:
	print('wrong dataset')
	exit()


# calculate information gain for each feature on the complete dataset
# for that, a tree classifier is instantiated
tree = cDTL.Learner()
tree.feature_types = feature_types
max_gain_per_axis, best_feature_vals = tree.get_split_gains(X,Y)

# order features by decreasing importance
order = np.flipud(np.argsort(max_gain_per_axis))
X = X[:,order]
feature_names = feature_names[order]
print(feature_names)

# split into training, testing, and validation sets
X_train,Y_train,X_test,Y_test,X_vali,Y_vali = dataHandler.split_sets(X, Y, ratio_train=.7, ratio_test=.2)


max_depth = 5

'''
In each iteration a complete training and testing process is done.
In each iteration the number of available features is decreased
'''
performances = []
for it in range(X.shape[1]):
	print(it)
	tree = cDTL.Learner(max_depth = max_depth, feature_names = feature_names)
	tree.learn(X_train[:,it:],Y_train,feature_types)
	Y_hat = [tree.predict(x) for x in X_test]
	perf = np.mean(Y_hat==Y_test)
	print(perf)
	performances.append(perf)




'''
do the same for a forrest
'''
performances2 = []
performances3 = []
performances4 = []
for it in range(X.shape[1]):
	print(it)
	forest = cDTL.ForestLearner(X_train[:,it:],Y,feature_types,
			feature_names=feature_names,
			n_trees = 200,
			batch_size = 100,
			max_depth = 1,
			random_splits = True)
	Y_hat = [forest.predict(x) for x in X_test]
	perf = np.mean(Y_hat==Y_test)
	print(perf)
	performances2.append(perf)
	Y_hat = [forest.predict_vote(x) for x in X_test]
	perf = np.mean(Y_hat==Y_test)
	print(perf)
	performances3.append(perf)
	Y_hat = [forest.predict_linear(x) for x in X_test]
	perf = np.mean(Y_hat==Y_test)
	print(perf)
	performances4.append(perf)


plt.plot(performances)
plt.plot(performances2)
plt.plot(performances3)
plt.plot(performances4)
plt.legend(['tree','forest','forest vote','forest linear'])
plt.show()

