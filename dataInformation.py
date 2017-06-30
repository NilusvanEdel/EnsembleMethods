import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


class DataInformation:
    def __init__(self, remove):
        self.remove = remove
        self.shroom = pd.read_csv('/home/nilus/PycharmProjects/EnsembleMethods/input/mushrooms.csv')
        # remove missing features and exclude veil-type (<-- only one feature included)
        if remove:
            self.shroom = self.shroom.drop(self.shroom[self.shroom['stalk-root'] == '?'].index)
            self.shroom = self.shroom.drop('veil-type', axis=1)

    def get_data(self):
        return self.shroom

    def get_one_hot(self):
        shroom = self.shroom
        if not self.remove:
            shroom = shroom.drop(shroom[shroom['stalk-root'] == '?'].index)
            shroom = shroom.drop('veil-type', axis=1)

        # transform data into one-hot vector
        lb = LabelBinarizer()
        for feature in shroom.columns:
            if len(shroom[feature].unique()) == 2:
                shroom[feature] = lb.fit_transform(shroom[feature])

        features_onehot = []
        for feature in shroom.columns[1:]:
            if len(shroom[feature].unique()) > 2:
                features_onehot.append(feature)
        temp = pd.get_dummies(shroom[features_onehot])
        shroom = shroom.join(temp)
        shroom = shroom.drop(features_onehot, axis=1)
        return shroom

    def get_list_of_significance(self):
        # returns a ordered list of the significance of each feature (from high to low) using the RandomForest of skikit
        shroom = self.shroom
        lbe = LabelEncoder()
        for feature in shroom.columns[1:]:
            shroom[feature] = lbe.fit_transform(shroom[feature])
        y = shroom['class'].values
        X = shroom.drop('class', axis=1).values
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1)
        rfc.fit(X, y)
        importances = rfc.feature_importances_
        features = shroom.columns[1:]
        sort_indices = np.argsort(importances)[::-1] #sort the array and start with the highest value
        sorted_features = []
        for idx in sort_indices:
            sorted_features.append(features[idx])
        # show figure
        '''
        plt.figure()
        plt.bar(range(len(importances)), importances[sort_indices], align='center');
        plt.xticks(range(len(importances)), sorted_features, rotation='vertical');
        plt.xlim([-1, len(importances)])
        plt.grid(False)
        plt.show()
        '''
        return sorted_features

