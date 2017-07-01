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
                if not(node.leaf):
                    self.build_tree(node)
                    print(node)
                    children = list(node.children.values())
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
                node.children = None

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

data = pd.read_csv("mushrooms\Dataset.data", delimiter=" ")
tree = DecisionTree(data)
