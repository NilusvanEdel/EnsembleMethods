import numpy as np
import pandas as pd
import math

class DecisionTree:

    root = None
    max_depth = "inf"

    def __init__(self, data_in, max_depth="inf"):
        data = None
        if(type(data_in) == str):
            data = pd.read_csv(data, delimiter=" ")
        elif(type(data_in) == pd.core.frame.DataFrame):
            data = data_in
        else:
            raise TypeError("wrong inpute type")
        self.max_depth=max_depth
        self.root = Node(data_in)

        todo = [self.root]
        depth = 0
        while(depth < float(self.max_depth)):
            tmp = []
            for node in todo:
                if not(node.leaf == True):
                    self.build_tree(node)
                    if not(node.leaf == True):
                        children = list(node.children.values())
                    else:
                        children = []
                    tmp += children
                else:
                    continue
            if(len(tmp) > 0):
                todo = tmp
                depth += 1
            else:
                break

    def build_tree(self, node):

        data = node.data
        information_gain = np.zeros(data.shape[1]-1)

        Y = data.groupby("class").size()
        total = Y.sum()
        E_unsplit = 0
        for value in set(Y.values):
            p = value/total
            E_unsplit -= p*math.log2(p)

        for i in range(1, data.shape[1]):
            df = data.iloc[:,[0,i]]
            df = df.groupby([df.iloc[:,0], df.iloc[:,1]]).size()

            E_feature = 0

            for j in df.index.levels[1]:
                df_j = df[:,j]
                p_j = df_j.sum() / total
                E_j = 0
                for value in df_j:
                    p = value / df_j.sum()
                    E_j -=  p*math.log2(p)
                E_feature += p_j*E_j

            information_gain[i-1] = E_unsplit - E_feature
            if(information_gain.max() > 0):
                key = np.where(information_gain == information_gain.max())[0][0]+1
                node.split_attribute_name = data.iloc[:,key].name
                label_set = list(set(data.iloc[:,key].values))
                split_data = [Node(data = pd.DataFrame(data[data.iloc[:,key] == label])) for label in label_set]
                split_dict = dict(zip(label_set, split_data))
                node.children = split_dict

            else:
                node.split_attribute_name = "None"
                node.leaf = True
                p_dict = {}
                s = data.iloc[:,0].value_counts().sum()
                for value in data.iloc[:,0].value_counts().index:
                    p_dict[value] = (data.iloc[0].value_counts()[value])/s
                node.children = p_dict
            node.data = None

    def classify_samples(self, samples):

        for i in range(len(samples)):
            split_node = self.root
            while(True):
                if(split_node.leaf == False):
                    key = split_node.split_attribute_name
                    label = samples.iloc[i][key]
                    split_node = split_node.children[label]
                else:
                    p_dict = split_node.children
                    samples.iloc[i][0] = max(p_dict, key=p_dict.get)
                    break
        return samples


class Node:

    leaf = False
    data = None
    split_attribute_name = "attribute"
    children = {}

    def __init__(self, data=None, **kwargs):
        self.data = data

        for key in kwargs:
            self.children[key] = kwargs[key]

    def __str__(self):
        return "#"+self.split_attribute_name+"#"

#######################################
#######################################
#######################################
df = pd.read_csv("Dataset.data", delimiter=" ")
A = np.zeros(10)
for i in range(10):

    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    test = df[~msk]
    validate = df[~msk]
    repl = {"class": {"p": "NA", "e": "NA"}}
    test = test.replace(repl)

    tree = DecisionTree(train)
    test = tree.classify_samples(test)

    ne = (test == validate).any(1)
    A[i] = ne.values.mean()
A = A*100
print(A)
