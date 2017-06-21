import numpy as np 
import csv
import continuousDTLearner as cDTL
import copy
import perceptron as p
import linear_classifier as lc
from batch import Batch
from scipy.misc import imread, imresize


# wdbc	['c']*X.shape[1]
# mushrooms
# min_tit3	['d','d','d','d','d','c','c']

def read_data(f_name = 'wdbc.csv'):
    file = open(f_name)
    reader = csv.reader(file)
    data = []
    for row in reader:
        data.append(row)
    data = np.array(data)
    Y = data[1:,0]
    code = np.unique(Y)
    Y = ((Y==code[0])+0.)*2-1
    X = data[1:,1:]
    # clear empty values
    idx = np.invert(np.any(X=='',1))
    X = X[idx,:]
    Y = Y[idx]
    feature_names = data[0,:]
    # shuffle data
    idx = np.random.permutation(Y.shape[0])
    Y = Y[idx]
    X = X[idx,:]
    file.close()
    return X, Y, feature_names,code

def read_image(f_name = 'test3.png'):
    data = imread(f_name)
    if len(data.shape)>2:
        data = data[:,:,0]
    data = np.array([(data[i,j],i,j) for i in range(data.shape[0]) for j in range(data.shape[1])])
    X = data[:,1:]
    Y = data[:,0]
    vals = np.unique(Y)
    code = np.arange(0,len(vals))
    for ix,v in enumerate(vals):
        Y[Y==v] = code[ix]
    Y = Y*2-1
    idx = np.random.permutation(Y.shape[0])
    Y=Y[idx]
    X=X[idx,:]
    return X,Y


def split_sets(X,Y,ratio_train=.4,ratio_test=.2):
	l = Y.shape[0]
	X_train = X[:int(l*ratio_train),:]
	X = X[int(l*ratio_train):,:]
	Y_train = Y[:int(l*ratio_train)]
	Y = Y[int(l*ratio_train):]
	X_test = X[:int(l*ratio_test),:]
	X = X[int(l*ratio_test):,:]
	Y_test = Y[:int(l*ratio_test)]
	Y = Y[int(l*ratio_test):]
	X_eval = X
	Y_eval = Y

	return X_train,Y_train,X_test,Y_test,X_eval,Y_eval


n_trees = 100
max_depth = 1
samples_per_tree = 20

min_perf_tree = .5
max_perf_tree = 1.0

ratio_train = .5
ratio_test = .25
ratio_eval = 1-(ratio_test+ratio_train)


X,Y,feature_names,code = read_data()

X,Y = read_image()

X_train,Y_train,X_test,Y_test,X_eval,Y_eval = split_sets(X,Y,
								ratio_train=ratio_train,ratio_test=ratio_test)
#X = np.hstack((X,np.random.normal(0,10,size=[X.shape[0],1])))

learners = []
train_batches = Batch(X_train,Y_train)
output_learners = []
perf_learners = []
for l in range(n_trees):
	x,y = train_batches.next(samples_per_tree)
	learner = cDTL.Learner(x,y,
		#['d','d','d','d','d','c','c'],
		['c']*X.shape[1],
		max_depth = max_depth)
	learners.append(learner)
	Y_hat = [learner.predict(x) for x in X_test]
	output_learners.append(Y_hat)
	perf = np.mean(Y_hat==Y_test) 
	perf_learners.append(perf)
	print('performance tree in test : {0}'.format(np.mean(Y_hat==Y_test)))


# delete bad performing learners
learners = [l for ix,l in enumerate(learners) if perf_learners[ix] > min_perf_tree and perf_learners[ix] < max_perf_tree]
output_learners = [o for ix,o in enumerate(output_learners) if perf_learners[ix] > min_perf_tree and perf_learners[ix] < max_perf_tree]
print('\n')

output_learners = np.array(output_learners)

linear_classifier = lc.LinearClassifier(output_learners.T,Y_test)

ensemble_tree = cDTL.Learner(output_learners.T,
							Y_test,
							['d']*output_learners.shape[0],
							max_depth = len(learners))


print()
output_learners = []
perf_learners = []
for learner in learners:
	Y_hat = [learner.predict(x) for x in X_eval]
	output_learners.append(Y_hat)
	perf = np.mean(Y_hat==Y_eval) 
	perf_learners.append(perf)
	print('performance tree in eval : {0}'.format(perf))

print('\n')


print('{0} trees of depth {1} in ensemble. '.format(len(learners),max_depth))
print('mean performance of trees	: {0}'.format(np.mean(perf_learners)))
print('performance best tree           : {0}'.format(np.max(perf_learners)))


output_learners = np.array(output_learners)
pred_lin=linear_classifier.predict(output_learners.T)
print('performance linear classifier 	: {0}'.format(np.mean(pred_lin==Y_eval)))


pred_ensemble_tree = np.array([ensemble_tree.predict(x) for x in output_learners.T])
print('performance ensemble tree    	: {0}'.format(np.mean(pred_ensemble_tree==Y_eval)))
