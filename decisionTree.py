import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import tree
import pydotplus

data = pd.read_csv("/home/nilus/PycharmProjects/EnsembleMethods/input/mushrooms.csv")


# convert the strings into integers to easier work on them
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])

# separate features and labels
X = data.iloc[:, 1:23]  # all rows, all the features and no labels
y = data.iloc[:, 0]  # all rows, label only

# separate the data intro train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

'''
# Standardise the Data to be between -1 and 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
'''

# predict the test data set with the decision tree
# max_depth=, max_features= are probably the most important ones to consider <-- how does max_features work?
clf = tree.DecisionTreeClassifier(max_depth=2, max_features=5)
clf = clf.fit(X_train, y_train)
pred = clf.predict(X_test)

# create a pdf with console: 'dot -Tpdf iris.dot -o iris.pdf' out of it <-- looks shitty and doesn't help
'''
with open("mushroom.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
'''
# count how many misclassifications happened
errors = 0
counter = 0
for no, cls in y_test.iteritems():
    if pred[counter] != cls:
        errors += 1
    counter += 1
print(errors, 'in ', len(pred), ' many test samples')





