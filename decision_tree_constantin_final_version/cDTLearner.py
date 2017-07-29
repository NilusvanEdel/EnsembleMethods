import numpy as np
import scipy.stats as stats
from math import log as mlog
import matplotlib.pyplot as plt
import string
from itertools import combinations




class LeafNode():

    def __init__(self,label,axis=None,val=None):
        self.label = label
        self.axis = axis
        self.val = val
        self.type = 'LN'


class DecisionNode():

    def __init__(self,axis=None,val=None):
        self.pos_child = None
        self.neg_child = None
        self.type = None
        self.axis = axis
        self.val = val
        self.ix = None

    def decide(self,x):
        if self.type == 'DDN':
            if self.val == x[self.axis]:
                return self.pos_child
            else:
                return self.neg_child
        elif self.type == 'CDN':
            if float(x[self.axis]) <= float(self.val):
                return self.pos_child
            else:
                return self.neg_child


    def split(self,X,Y):
        if self.type == 'DDN':
            ix = X[:,self.axis]==self.val
            jx = np.invert(ix)
            pos_X = X[ix]
            neg_X = X[jx]
            pos_Y = Y[ix]
            neg_Y = Y[jx]
        elif self.type == 'CDN':
            self.val = float(self.val)
            x = np.array([np.float(i) for i in X[:,self.axis]])
            ix = x <= self.val
            jx = x > self.val
            pos_X = X[ix]
            neg_X = X[jx]
            pos_Y = Y[ix]
            neg_Y = Y[jx]
        self.ix = ix 
        return pos_X, pos_Y, neg_X, neg_Y, ix, jx


class ForestLearner():

    def __init__(self, X, Y, feature_types, feature_names = None,
                 n_trees = 12, max_depth = 10, features_per_tree = None,
                 random_splits = False, batch_size = None,
                 share_features = False, ratio_train = 0.7):

        self.trees = []
        self.n_trees = n_trees
        self.feature_names = feature_names
        self.max_depth = max_depth
        self.feature_types = feature_types
        self.share_features = share_features
        self.random_splits = random_splits
        if not batch_size is None:
            self.batch_size = batch_size
        else:
            self.batch_size = int(X.shape[0]/self.n_trees)*2

        # split training testing set
        idx = np.random.permutation(X.shape[0])
        X = X[idx,:]
        Y = Y[idx]
        t_idx = int(ratio_train*X.shape[0])
        X_train = X[:t_idx,:]
        Y_train = Y[:t_idx]
        X_test = X[t_idx:,:]
        Y_test = Y[t_idx:]

        
        if self.share_features:
            if features_per_tree is None:
                self.features_per_tree = X.shape[1]
            else:
                self.features_per_tree = features_per_tree
            self.feature_indices_list = [np.random.randint(X.shape[1],size = self.features_per_tree) for t in range(self.n_trees)]
        else:
            if features_per_tree is None:
                self.features_per_tree = X.shape[1]
            else:
                self.features_per_tree = features_per_tree
            if X.shape[1]/self.features_per_tree < 1:
                self.n_trees = X.shape[1]
                self.features_per_tree = 1
            self.feature_indices_list = []
            tmp_list = list(range(X.shape[1]))
            for t in range(self.n_trees):
                self.feature_indices_list.append(np.array(tmp_list[:self.features_per_tree]))
                tmp_list = tmp_list[self.features_per_tree:]


        self.trees = []
        self.output_trees = []
        for t in range(n_trees):
            tree = Learner(max_depth=self.max_depth,
                           random_splits = self.random_splits, 
                           feature_names = self.feature_names,
                           feature_indices = self.feature_indices_list[t])
            X_train_batch,Y_train_batch = self.batch(X_train,Y_train,size = self.batch_size)
            tree.learn(X_train_batch,Y_train_batch,self.feature_types)
            Y_hat = [tree.predict(x) for x in X_test]
            self.output_trees.append(Y_hat)
            self.trees.append(tree)
        self.output_trees = np.array(self.output_trees).T
        self.forest = Learner(max_depth = self.n_trees)
        self.forest.learn(self.output_trees,Y_test,'d'*self.n_trees)


        #tmp = np.array((np.mean(self.output_trees,1)>0.)*2-1)
        tmp_inv = np.linalg.pinv(self.output_trees)
        self.x_out = np.dot(tmp_inv,Y_test)


    def predict(self,x):
        self.output_trees = []
        for tree in self.trees:
            self.output_trees.append(tree.predict(x))
        self.output_trees = np.array(self.output_trees)
        return self.forest.predict(self.output_trees)

    def predict_linear(self,x):
        output_trees = []
        for tree in self.trees:
            output_trees.append(tree.predict(x))
        output_trees = np.array(output_trees)

        return (np.dot(output_trees.T,self.x_out)>0.)*2-1

    def predict_vote(self,x):
        output_trees = []
        for tree in self.trees:
            output_trees.append(tree.predict(x))
        output_trees = np.sign(np.mean(output_trees))

        return output_trees

    def batch(self,X,Y,size):
        idx = np.random.randint(X.shape[0],size=[size])
        return X[idx,:],Y[idx]



class Learner():
    # feature_types is a list of the same length as one data point
    # containing either a 'd' for categorical, or 'c' for continuous
    # values. e.g. ['c']*X.shape[1] 
    def __init__(self, max_depth = 3, random_splits = False, feature_names=None, feature_indices = None):
        self.max_depth = max_depth
        self.random_splits = random_splits
        self.feature_indices = feature_indices
        self.feature_names = feature_names



    def learn(self,X,Y,feature_types):
        if self.feature_names is None:
            self.feature_names = self.code_features(X)
        if not self.feature_indices is None:
            idx = [any(self.feature_indices == ix) for ix in range(X.shape[1])]
            idx = np.invert(idx)
            X[:,idx] = '0'
        else:
            self.feature_indices = np.arange(0,X.shape[1],1)
        self.feature_types = feature_types
        self.root = DecisionNode()
        self.build_tree(X,Y, self.root,0)

    def code_features(self,X):
        if X.shape[1]<=len(string.ascii_lowercase):
            code = string.ascii_lowercase[:X.shape[1]]
        else:
            code = list(combinations(string.ascii_lowercase, 3))[:X.shape[1]]
            code = [c[0]+c[1]+c[2] for c in code]
        return code

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
        if Y.shape[0] == 0:
            return 0
        if all(Y==Y[0]):
            return 0
        labels = np.unique(Y)
        label_probs = []
        for l in labels:
            label_probs.append(np.mean(Y==l))
        entropy = 0
        for p in label_probs:
            #entropy -= p * mlog(p, len(labels))
            entropy -= p * mlog(p, 2) # since we only classify one class
        return entropy


    def information_gain(self,X,Y,axis):
        # first, calculate global entropy
        global_entropy = self.entropy(Y)
        # for each possible given class label, calculate the entropy
        # multiply by chance to get label
        labels = np.unique(X)
        cum_entropy = 0
        #label_gain = []
        entropy_after_split = []
        if self.feature_types[axis] == 'd':
            for label in labels:
                p_label = np.mean(X==label)
                p_not_label = 1 - p_label
                entropy_label = self.entropy(Y[X==label])
                entropy_not_label = self.entropy(Y[X!=label])
                entropy_after_split.append(entropy_label*p_label + entropy_not_label*p_not_label)
        elif self.feature_types[axis] == 'c':
            x = np.array([float(x) for x in X])
            labels = np.unique(x)
            for label in labels:
                p_label = np.mean(x<=label)
                p_not_label = 1 - p_label
                entropy_label = self.entropy(Y[x<=label])
                entropy_not_label = self.entropy(Y[x>label])
                entropy_after_split.append(entropy_label*p_label + entropy_not_label*p_not_label)
        cum_entropy = np.min(entropy_after_split)
        gain = global_entropy - cum_entropy
        best_label = labels[np.argmin(entropy_after_split)] 
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
            inf_gain, label = self.information_gain(x,Y,axis)
            max_gain_per_axis.append(inf_gain)
            best_feature_vals.append(label)
        return max_gain_per_axis, best_feature_vals


    def get_split(self,X,Y):
        split_gains, split_val = self.get_split_gains(X,Y)
        axis = np.argmax(split_gains)
        val = split_val[axis]
        return axis, val


    def build_tree(self,X,Y,cur_node,depth):
        if not self.random_splits:
            axis, val = self.get_split(X,Y)
        else:
            axis = self.feature_indices[np.random.randint(len(self.feature_indices))]
            val = np.unique(X[:,axis])
            val = val[np.random.randint(val.shape[0])]
        # give parameters to node
        cur_node.axis = axis
        cur_node.val = val
        if self.feature_types[axis] == 'c':
            cur_node.type = 'CDN'
        elif self.feature_types[axis] == 'd':
            cur_node.type = 'DDN'
        # apply split
        pos_X, pos_Y, neg_X, neg_Y,ix,jx = cur_node.split(X,Y)
        # TODO: build something for the active weights subset
        if pos_Y.shape[0]==0:
            cur_node.neg_child = LeafNode(self.most_frequent_label(neg_Y))
            cur_node.pos_child = LeafNode(self.opposite(cur_node.neg_child.label))
            return
        if neg_Y.shape[0]==0:
            cur_node.pos_child = LeafNode(self.most_frequent_label(pos_Y))
            cur_node.neg_child = LeafNode(self.opposite(cur_node.pos_child.label))
            return
        if depth >= self.max_depth-1:
            cur_node.pos_child = LeafNode(self.most_frequent_label(pos_Y))
            cur_node.neg_child = LeafNode(self.most_frequent_label(neg_Y))
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
    



    def predict(self,x):
        self.n = self.root
        depth = 0
        while not type(self.n) == LeafNode:
            self.n = self.n.decide(x)
            depth+=1
        return self.n.label





    '''
    Only printing methods from here on

    '''
    def print2d(self,X,Y):
        '''
        If data only has two different feature axes, 
        this method can plot the decision boundaries 
        in two 
        '''
        x_min = np.min(X[:,0])-1
        x_max = np.max(X[:,0])
        y_min = np.min(X[:,1])-1
        y_max = np.max(X[:,1])
        ax = [[x_min,x_max],[y_min,y_max]]
        node = self.root
        self.p2d(node, '+', ax)
        plt.show()

    def p2d(self,node,type,ax):
        if node.type == 'LN':
            return
        axp = np.copy(ax)
        axn = np.copy(ax)
        if node.axis == 0:
            axp[1][1] = node.val
            axn[1][0] = node.val
        if node.axis == 1:
            axp[0][1] = node.val
            axn[0][0] = node.val
        if node.axis == 0:
            plt.plot(ax[0],[node.val,node.val],'w',linewidth = 3)
            if node.pos_child.type == 'LN':
                    plt.text(np.mean(ax[0]),np.mean([node.val,ax[1][0]]),node.pos_child.label,fontsize = 10,color = 'r',va = 'center',ha = 'center')
            if node.neg_child.type == 'LN':
                    plt.text(np.mean(ax[0]),np.mean([node.val,ax[1][1]]),node.neg_child.label,fontsize = 10,color = 'r',va = 'center',ha = 'center')
        if node.axis == 1:
            plt.plot([node.val,node.val],ax[1],'w',linewidth = 3)
            if node.pos_child.type == 'LN':
                    plt.text(np.mean([node.val,ax[0][0]]),np.mean(ax[1]),node.pos_child.label,fontsize = 10,color = 'r',va = 'center',ha = 'center')
            if node.neg_child.type == 'LN':
                    plt.text(np.mean([node.val,ax[0][1]]),np.mean(ax[1]),node.neg_child.label,fontsize = 10,color = 'r',va = 'center',ha = 'center')
        self.p2d(node.pos_child, '+', axp)
        self.p2d(node.neg_child, '-', axn)



    def print_tree(self):
        '''
        Prints decision tree structure
        '''
        print('----===='*10)
        cur_node = self.root
        self.recursive_print(cur_node,'root',-1)
        print('----===='*10)


    def recursive_print(self,node,mode,indent):
        indent += 1
        lead = '    '*indent
        if not type(node)==LeafNode:
            print(lead+'+-----------'.format(indent))
            print(lead+'| '+ node.type)
            print(lead+'| level: {0}'.format(indent))
            print(lead+'| type:  '+mode)
            print(lead+'| name:  '+self.feature_names[node.axis])
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
