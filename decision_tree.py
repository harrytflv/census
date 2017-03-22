import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import csv
import numpy as np
import scipy
import scipy.io as sio
import itertools
import operator
import scipy.stats
from collections import Counter
from numpy import genfromtxt

import pdb

MAX_HEIGHT = 100
MIN_LEAF = 5
DEBUGGING = False
DATASET = "Census"
ITERS = 100
FOREST = True

def shrink(size):
    if (DEBUGGING):
        return size // 120, size // 20
    else:
        return size // 6, size

def judge(number):
	if (number > 0.5):
		return 1
	else:
		return 0
judge = np.vectorize(judge)

class Node:
    def __init__(self):
        self.split_rule = (None, None, None)
        self.data_points = []
        self.left = None
        self.right = None
        self.label = None

    def set_leaf(self, data, labels):
        self.data_points = train_data
        self.label = Counter(labels).most_common(1)[0][0]

class DecisionTree():
    def __init__(self, max_height = MAX_HEIGHT):
        self.root = Node()
        self.max_height = max_height

    def train(self, train_data, train_labels, weights = np.zeros((1,1))):
        if (weights.shape == (1,1)):
            weights = np.ones(train_data.shape[0]) / train_data.shape[0]
        self._train(train_data, train_labels, self.root, weights, 0)

    def _train(self, train_data, train_labels, node, weights, height):

        # if (height < 3):
        #     print("At height: {0}".format(height))

        # When there is only one training data, or only one label, or exceeding max_height
        # Define this node as a leaf
        if train_data.shape[0] < MIN_LEAF or height >= self.max_height or len(set(train_labels)) == 1:
            # print("Leaf height: {0}".format(height))
            node.set_leaf(train_data, train_labels)

        # Else try to separate the data
        else:
            node.left = Node()
            node.right = Node()
            node.split_rule = self.segmeter(train_data, train_labels, weights)

            # When it turned out that the data is not separable
            if (node.split_rule == (-1, -1, False)):
                # Print the leaf height.
                # print("Leaf height: {0}".format(height))
                node.set_leaf(train_data, train_labels)

            else:
                l_cond = train_data[:,node.split_rule[0]] <= node.split_rule[1]
                r_cond = train_data[:,node.split_rule[0]] > node.split_rule[1]
                nan_cond = np.isnan(train_data[:,node.split_rule[0]])
                train_data_l = train_data[l_cond]
                train_data_r = train_data[r_cond]
                train_labels_l = train_labels[l_cond]
                train_labels_r = train_labels[r_cond]
                weights_l = weights[l_cond]
                weights_r = weights[r_cond]

                # Treat nan as mode
                # If mode is greater than threshold, then the nan data are put to right
                train_data_nan = train_data[nan_cond]
                train_labels_nan = train_labels[nan_cond]
                weights_nan = weights[nan_cond]
                if (node.split_rule[2]):
                    train_data_r = np.r_[train_data_r, train_data_nan]
                    train_labels_r = np.r_[train_labels_r, train_labels_nan]
                    weights_r = np.r_[weights_r, weights_nan]
                else:
                    weights_l = np.r_[weights_l, weights_nan]
                    train_data_l = np.r_[train_data_l, train_data_nan]
                    train_labels_l = np.r_[train_labels_l, train_labels_nan]


                self._train(train_data_l, train_labels_l, node.left, weights_l, height + 1)
                self._train(train_data_r, train_labels_r, node.right, weights_r, height + 1)

    def predict(self, test_data):
        predictions = np.zeros(test_data.shape[0])
        for i in range(0, test_data.shape[0]):
            predictions[i] = self._predict(test_data[i], self.root)
        return predictions

    def _predict(self, data_point, node):
        if (node.label != None):
            return node.label
        else:
            if data_point[node.split_rule[0]] > node.split_rule[1] or (np.isnan(data_point[node.split_rule[0]]) and node.split_rule[2]):
                return self._predict(data_point, node.right)
            else:
                return self._predict(data_point, node.left)

    # Calculate how bad a split is
    def impurity(self, left_labels, left_weights, right_labels, right_weights):
        return (np.sum(left_weights) * self.entropy(left_labels) + \
                np.sum(right_weights) * self.entropy(right_labels)) / \
               (np.sum(left_weights) + np.sum(right_weights))

    # Calculate the entropy of an array of labels
    def entropy(self, labels):
        if labels.shape[0] == 0:
            return 0
        counts = np.array(Counter(labels).most_common())[:,1]
        entropy = scipy.stats.entropy(counts)
        return entropy

    # Find the best split for data and labels
    def segmeter(self, data, labels, weights):
        max_info_gain = 0
        max_split_rule = (-1, -1, False)
        max_split_rule_mode = np.nan
        modes = scipy.stats.mode(data)[0][0]

        ratio = (np.sqrt(data.shape[1]) + 1) / data.shape[1]
        on_cnt = int(data.shape[1] * ratio)
        switch = np.array(([1] * on_cnt) + [0] * (data.shape[1] - on_cnt))
        np.random.shuffle(switch)

        # Find the split that maximize infomation gain
        # For each feature
        for feature in range(0, data.shape[1]):

            if not switch[feature]:
                continue

            values = data[:, feature]
            mode = modes[feature]
            dist_values = np.unique(values)
            dist_values = dist_values[~np.isnan(dist_values)]



            # For each possible split point
            for value in dist_values:


                left_labels = labels[data[:, feature] <= value]
                left_weights = weights[data[:, feature] <= value]
                right_labels = labels[data[:, feature] > value]
                right_weights = weights[data[:, feature] > value]

                # Treat nan as mode of this feature.
                # If mode(which was nan) are greater than threshold, then put it to the right
                nan_labels = labels[np.isnan(data[:,feature])]
                nan_weights = weights[np.isnan(data[:,feature])]
                if (mode > value):
                    right_labels = np.r_[right_labels, nan_labels]
                    right_weights = np.r_[right_weights, nan_weights]
                else:
                    left_labels = np.r_[left_labels, nan_labels]
                    left_weights = np.r_[left_weights, nan_weights]

                # The value is max, cannot separate
                if (left_labels.shape[0] == 0 or right_labels.shape[0] == 0):
                    info_gain = 0
                else:
                    info_gain = self.entropy(labels) - self.impurity(left_labels, left_weights, right_labels, right_weights)

                if (info_gain > max_info_gain):
                    max_split_rule = (feature, value, mode > value)
                    max_info_gain = info_gain

        return max_split_rule

    def print_root(self):
        print(self.root.split_rule)

class Forest():
    def __init__(self, iters = ITERS):
        self.trees = []
        self.weights = []
        self.iters = iters

    def train(self, train_data, train_labels):
        self.alphas = np.ones(self.iters)
        self.weights = np.ones(train_data.shape[0]) / train_data.shape[0]
        for t in range(0, self.iters):
            cur_tree = DecisionTree()
            print("iter: {0}".format(t))
            # indices = np.random.rand((train_data.shape[0])) > 0.5
            # cur_train_data = train_data[indices]
            # cur_train_labels = train_labels[indices]
            # cur_weights = self.weights[indices]
            # pdb.set_trace()
            cur_tree.train(train_data, train_labels, self.weights)
            train_pred = cur_tree.predict(train_data)
            e = metrics.accuracy_score(train_labels, train_pred)
            self.alphas[t] = 0.5 * np.log(e / (1 - e))
            for i in range(0, train_data.shape[0]):
                indicator = int(train_labels[i] == train_pred[i])
                self.weights[i] = self.weights[i] * np.exp( -self.alphas[t] * indicator)
            self.weights = self.weights / np.sum(self.weights)
            self.trees.append(cur_tree)
            cur_tree.print_root()

    def predict(self, test_data):
        predictions = np.zeros(test_data.shape[0])
        self.alphas = self.alphas / np.sum(self.alphas)
        for i in range(0, self.iters):
            predictions += self.trees[i].predict(test_data) * self.alphas[i]
        return judge(predictions)


def load_dataset(dataset = DATASET):
    if (dataset == "Census"):
        filename = 'data/census_data/census_data.mat'
    else:
        filename = 'data/spam_data/spam_data.mat'
    data = sio.loadmat(filename)
    train_data = data['training_data']
    train_labels = data['training_labels'][0]
    test_data = data['test_data']

    rng_state = np.random.get_state()
    np.random.shuffle(train_data)
    np.random.set_state(rng_state)
    np.random.shuffle(train_labels)

    val_size, train_size = shrink(train_data.shape[0])
    val_data, val_labels = train_data[0:val_size], train_labels[0:val_size]
    train_data, train_labels = train_data[val_size:train_size], train_labels[val_size:train_size]

    return train_data, train_labels, val_data, val_labels, test_data

train_data, train_labels, val_data, val_labels, test_data = load_dataset()
if (FOREST):
	classifier = Forest()
else:
	classifier = DecisionTree()

classifier.train(train_data, train_labels)
train_pred = classifier.predict(train_data)
val_pred = classifier.predict(val_data)

print("Train Accuracy: {0}".format(metrics.accuracy_score(train_labels, train_pred)))
print("Validation Accuracy: {0}".format(metrics.accuracy_score(val_labels, val_pred)))

test_pred = classifier.predict(test_data)
id = np.array(list(range(1,len(test_pred) + 1)))
output = np.array([id,test_pred]).T
np.savetxt(DATASET + ".csv", output, delimiter = ',')
