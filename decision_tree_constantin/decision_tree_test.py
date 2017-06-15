import numpy as np 
import csv
import DTLearner as DTL
import copy
import perceptron as p
import linear_classifier as lc



def read_data(f_name = 'mushrooms.csv'):
    file = open(f_name)
    reader = csv.reader(file)
    data = []
    for row in reader:
        data.append(row)
    data = np.array(data)
    Y = data[1:,0]
    Y = ((Y=='e')+0.)*2-1
    X = data[1:,1:]
    feature_names = data[0,:]
    file.close()
    return X, Y, feature_names


# get data
X,Y,feature_names = read_data()

# numer of weak learners in ensemble
no_weak = 5

# create training/testing/evaluation sets
ratio_train = .6
ratio_eval = .2
ratio_test = 1-(ratio_train+ratio_eval)

batches = DTL.Batch([X,Y])

train_batch = DTL.Batch(batches.next(int(ratio_train*len(Y))))
test_batch = DTL.Batch(batches.next(int(ratio_test*len(Y))))
eval_X,eval_Y = batches.next(int(ratio_eval*len(Y)))

test_batch_copy = copy.deepcopy(test_batch)

learners = []
perf_learners = []
for lx in range(no_weak):
    x,y = train_batch.next(int(train_batch.X.shape[0]/no_weak))
    learner = DTL.Learner()
    learner.init_tree(x,y)

    x_test,y_test = test_batch.next(int(test_batch.X.shape[0]/no_weak))
    y_pred = [learner.predict(x) for x in x_test]

    learners.append(learner)
    perf = np.mean(y_pred==y_test)
    perf_learners.append(perf)
    print('L{1} |correct predicted: {0}'.format(perf,lx))

learners = [l for ix,l in enumerate(learners) if perf_learners[ix] > .6]


perc_train = []
perc_targets = []

for ix in range(int(ratio_test*len(Y))):
    x,y = test_batch_copy.next(1)
    preds = []
    for learner in learners:
        pred = learner.predict(x[0])
        preds.append(pred)
    perc_train.append(preds)
    perc_targets.append(y)

perc_train = np.array(perc_train)*2-1
perc_targets = np.squeeze(np.array(perc_targets)*2-1)



##
# Train Linear Classifier on output of tree ensemble
linear_classifier = lc.LinearClassifier(perc_train,perc_targets)
##
# Train simple Perceptron on output of tree ensemble
P = p.Perceptron(perc_train,perc_targets)






## TEST Linear Classifier and Perceptron on Evaluation Data:

# first, run eval data through trees and collect their output:
outp_trees = []
perf_trees = []
lx = 0
for learner in learners:
    outp_learner = []
    for x in eval_X:
        pred = learner.predict(x)
        outp_learner.append(pred)
    outp_trees.append(outp_learner)
    # calc performance of single tree
    perf = np.mean(np.sign(outp_learner)==np.sign(eval_Y))
    perf_trees.append(perf)
    print('L{0} | correct in evalutation: {1}'.format(lx,perf))
    lx+=1

outp_trees = np.array(outp_trees).T

print('-'*10)
print('best performing single tree: {0}'.format(np.max(perf_trees)))


pred_lin=linear_classifier.predict(outp_trees)
print('performance linear classifier: {0}'.format(np.mean(pred_lin==eval_Y)))


pred_perc=P.predict(outp_trees)
print('performance perceptron: {0}'.format(np.mean(pred_perc==eval_Y)))
print()
