import numpy as np
import finalDataHandler as dataHandler
import cDTLearner as cDTL
import fern as fern
import matplotlib.pyplot as plt 
from os import listdir

# 'wdbc.csv' / 'mushrooms.csv'
data_set_name = 'mushrooms.csv'

# load data
X, Y, feature_names, codeing = dataHandler.read_data(f_name = data_set_name)

if data_set_name == 'wdbc.csv':
	feature_types = ['c']*X.shape[1]
elif data_set_name == 'mushrooms.csv':
	feature_types = ['d']*X.shape[1]
else:
	print('wrong dataset')
	exit()



# file_name = 'forest'
file_name = 'fern'
extension = '_depth_'+data_set_name.split('.')[0]


'''
please put our respective learner inside this loop.
Save the training results.


here we want to iterate over several depths for the tree,
number of splits for the fern, or number of learners for boosting.
'''

performances = []

max_depth = 20


depth_spacing = np.arange(1,max_depth,1)
for depth in depth_spacing:
	perf = []
	print('depth: {0}'.format(depth))
	for iteration in range(20):
		# split into training, testing, and validation sets
		X_train,Y_train,X_test,Y_test,X_vali,Y_vali = dataHandler.split_sets(X, Y, ratio_train=.7, ratio_test=.3, random_seed = iteration)
		# PUT YOUR LEARNER HERE
		'''
		tree = cDTL.Learner(max_depth = depth, feature_names = feature_names)
		tree.learn(X_train,Y_train, feature_types)
		Y_hat = [tree.predict(x) for x in X_test]
		'''

		# ferns
		if (data_set_name == 'mushrooms.csv'):
			ferny = fern.Fern(X_train, Y_train, depth, continuous=False)
		else:
			ferny = fern.Fern(X_train, Y_train, depth, continuous=True)
		Y_hat = ferny.pred(X_test, Y_test)
		'''
		forest 
		'''
		'''
		n_trees = depth
		forest = cDTL.ForestLearner(X,Y, feature_types, 
				n_trees = n_trees, max_depth = 1, ensemble_tree_depth = int(np.log2(n_trees)),
				batch_size = 5,
				feature_names = feature_names)
		Y_hat = [forest.predict(x) for x in X_test]
		'''
		perf.append(np.mean(Y_hat==Y_test))
		print('		iteration: {0} | performance: {1}'.format(iteration+1,np.mean(Y_hat==Y_test)))
	performances.append(np.mean(perf))
	print('mean of depth ', depth, ' : ', np.mean(perf))


result = np.vstack((depth_spacing,performances))

if file_name+extension in [l.split('.')[0] for l in listdir()]:
	print('denk dir n anderen dateinamen aus >:( ')
else:
	np.save(file_name+extension, result)


plt.plot(performances)
plt.show()

