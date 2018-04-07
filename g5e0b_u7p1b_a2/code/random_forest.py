from random_tree import RandomTree
from decision_tree import DecisionTree
from scipy import stats

import numpy as np
import os
import pickle


def load_dataset(filename):
    with open(os.path.join('..', 'data', filename), 'rb') as f:
        return pickle.load(f)

class RandomForest:
    def __init__(self, num_trees, max_depth):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.list_of_trees = []

    def fit(self, X, y):
        numTrees = self.num_trees
        list_of_trees = []

        for x in range(numTrees):
            tree = RandomTree(max_depth=self.max_depth)
            tree.fit(X, y)
            list_of_trees.append(tree)

        self.list_of_trees = list_of_trees

    def predict(self, X):
        list_of_trees = self.list_of_trees

        list_of_ys = []

        for x in list_of_trees:
            y = x.predict(X)
            list_of_ys.append(y)

        mode = stats.mode(list_of_ys)
        to_Return_Mode = mode[0]

        return to_Return_Mode


    @staticmethod
    def evaluate_model(model):
        dataset = load_dataset('vowel.pkl')
        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']
        print("\nn = %d, d = %d\n" % X.shape)

        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print("    Training error: %.3f" % tr_error)
        print("    Testing error: %.3f" % te_error)


