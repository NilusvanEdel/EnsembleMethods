from __future__ import division
import numpy as np
import scipy.stats as stats
from math import log as mlog
import matplotlib.pyplot as plt


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

        return pos_X, pos_Y, neg_X, neg_Y


class DecisionTree():

    def __init__(self,axis,val):
        self.root = root
        self.val = root.val
        self.axis = root.axis
        self.type = 'DN'


class Learner():
	# feature_types is a list of the same length as one data point
	# containing either a 'd' for discrete, or 'c' for continuous
	# values. e.g. ['c']*X.shape[1] 
    def __init__(self,X,Y,feature_types,feature_names=None,max_depth = 3,beta = None):
        print("created Learner")
        self.X = X
        self.Y = Y
        self.feature_types = feature_types
        self.feature_names = feature_names
        self.max_depth = max_depth
        if beta == None:
            self.beta = np.ones(X[0].size)
        else:
            self.beta = beta
        self.init_tree(self.X,self.Y)
        
    def init_tree(self,X,Y):
        axis, val = self.get_split(X,Y)
        root = DecisionNode()
        if self.feature_types[axis] == 'c':
            root.type = 'DDN'
        elif self.feature_types[axis] == 'd':
            root.type = 'CDN'
        self.root = root 
        self.build_tree(X,Y,self.root,0)


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

            #label_probs.append(np.mean(Y==l))
            cntSum = 0
            for i,y in enumerate(Y):
                if y == l:
                    cntSum += self.beta[i]
            label_probs.append(cntSum / np.sum(self.beta))
                    
        entropy = 0
        for p in label_probs:
            entropy -= p * mlog(p, len(labels))
        return entropy

    def card_var(self,Y):
        if Y.shape[0]==0:
            return 0
        s = 0
        for j in range(Y.shape[0]):
            for i in range(Y.shape[0]):
                s += .5*(Y[i]-Y[j])**2
        s = (1/Y.shape[0]**2) * s 
        return s


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
                #p_label = np.mean(X==label)
                #per_label = self.entropy(Y[X==label]) 
                #cum_entropy += per_label * p_label
                #label_gain.append(per_label)
                p_label = np.mean(X==label)
                p_not_label = 1 - p_label

                entropy_label = self.entropy(Y[X==label])
                entropy_not_label = self.entropy(Y[X!=label])

                entropy_after_split.append(entropy_label*p_label + entropy_not_label*p_not_label)

        elif self.feature_types[axis] == 'c':
            x = np.array([float(x) for x in X])
            #labels = np.linspace(np.min(x),np.max(x),20)
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
        # find good split axis
        axis, val = self.get_split(X,Y)
        # give parameters to node
        cur_node.axis = axis
        cur_node.val = val
        if self.feature_types[axis] == 'c':
            cur_node.type = 'CDN'
        elif self.feature_types[axis] == 'd':
            cur_node.type = 'DDN'
        # apply split
        pos_X, pos_Y, neg_X, neg_Y = cur_node.split(X,Y)
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
            cur_node.neg_child = LeafNode(self.opposite(cur_node.pos_child.label))
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





    
    def recursive_print(self,node,mode,indent):
        indent += 1
        lead = '    '*indent

        if not type(node)==LeafNode:
            print(lead+'+-----------'.format(indent))
            print(lead+'| '+ node.type)
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
        print('----===='*10)
        cur_node = self.root
        self.recursive_print(cur_node,'root',-1)
        print('----===='*10)


    def predict(self,x):
        self.n = self.root
        depth = 0
        while not type(self.n) == LeafNode:
            self.n = self.n.decide(x)
            depth+=1
        return self.n.label

    def print2d(self):
        #ix = self.Y == self.Y[0]
        #plt.plot(self.X[ix,0],self.X[ix,1],'r*')
        #ix = np.invert(ix)
        #plt.plot(self.X[ix,0],self.X[ix,1],'b*')
        x_min = np.min(self.X[:,0])-1
        x_max = np.max(self.X[:,0])
        y_min = np.min(self.X[:,1])-1
        y_max = np.max(self.X[:,1])

        ax = [[x_min,x_max],[y_min,y_max]]

        node = self.root
        if node.axis == 0:
            plt.plot([x_min,x_max],[node.val,node.val],'k',linewidth = 3)
        if node.axis == 1:
            plt.plot([node.val,node.val],[y_min,y_max],'k',linewidth = 3)


        axp = np.copy(ax)
        axn = np.copy(ax)

        if node.axis == 0:
            axp[1][1] = node.val
            axn[1][0] = node.val
        if node.axis == 1:
            axp[0][1] = node.val
            axn[0][0] = node.val


        self.p2d(node.pos_child, '+', axp)
        self.p2d(node.neg_child, '-', axn)

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
            plt.plot(ax[0],[node.val,node.val],'k',linewidth = 3)
        if node.axis == 1:
            plt.plot([node.val,node.val],ax[1],'k',linewidth = 3)


        self.p2d(node.pos_child, '+', axp)
        self.p2d(node.neg_child, '-', axn)
