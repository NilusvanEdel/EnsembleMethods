import numpy as np
import finalDataHandler as dataHandler
#import cDTLearner as cDTL 
import matplotlib.pyplot as plt 
from ensembleMethods import BaggedLearner
from ensembleMethods import AdaBoost
from os import listdir

# 'wdbc.csv' / 'mushrooms.csv'
data_set_name = 'wdbc.csv'

# load data
X, Y, feature_names, codeing = dataHandler.read_data(f_name = data_set_name)

if data_set_name == 'wdbc.csv':
    feature_types = ['c']*X.shape[1]
elif data_set_name == 'mushrooms.csv':
    feature_types = ['d']*X.shape[1]
else:
    print('wrong dataset')
    exit()


# split into training, testing, and validation sets
X_train,Y_train,X_test,Y_test,X_vali,Y_vali = dataHandler.split_sets(X, Y, ratio_train=.7, ratio_test=.3)




'''
please put our respective learner inside this loop.
Save the training results.

before computing the effect of noise, please find optimal 
parameters for your learner.

'''


file_name = 'stumpLearner'
extension = '_noise_'+data_set_name.split('.')[0]

performances = []

noise_spacing = np.logspace(0,1,10)-1

for l in noise_spacing:
    perf = []
    print('noise level: {0}'.format(l))
    for iteration in range(10):
        # PUT YOUR LEARNER HERE 
        X_train_noisy = dataHandler.noise(X_train, level = l, random_seed = iteration)
        '''
        tree
        '''
        
        learner = AdaBoost(X_train_noisy, Y_train,ensSize=1)
        #learner = BaggedLearner(X_train_noisy, Y_train,feature_types,feature_names,max_depth = 1,ensSize=19,random_splits = True)
        #tree.learn(X_train_noisy,Y_train, feature_types)
        Y_hat = [learner.predict(x) for x in X_test]
        
        '''
        forest 
        '''
        '''
        n_trees = 1000
        forest = cDTL.ForestLearner(X,Y, feature_types, 
                n_trees = n_trees, max_depth = 1, ensemble_tree_depth = 10,
                batch_size = 20,
                feature_names = feature_names)
        Y_hat = [forest.predict(x) for x in X_test]
        '''

        perf.append(np.mean(Y_hat==Y_test))
        print('        iteration: {0} | performance: {1}'.format(iteration+1,np.mean(Y_hat==Y_test)))
    performances.append(np.mean(perf))

performances = np.array(performances)

result = np.vstack((noise_spacing,performances))


if file_name+extension in [l.split('.')[0] for l in listdir()]:
    print('denk dir n anderen dateinamen aus >:( ')
else:
    np.save(file_name+extension, result)


plt.plot(performances)
plt.show()

