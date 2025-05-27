import numpy as np

ATTRIBUTES = ['name', 'color', 'format']

class DecisionTree:
    def __init__(self, X, y, threshold=1.0, max_depth=None): # Additional optional arguments can be added, but the default value needs to be provided
        self.is_leaf = False
        self.label = None
        self.attribute = None
        self.branches = {}

        if len(set(y)) == 1:
            self.is_leaf = True
            self.label = y[0]
            return

        if len(X[0]) == 0 or (max_depth is not None and max_depth == 0):
            self.is_leaf = True
            self.label = np.sign(sum(y)) if y else 1
            return

        best_attr_index, best_gain = self.choose_best_attribute(X, y)

        if best_gain < threshold:
            self.is_leaf = True
            self.label = np.sign(sum(y)) if y else 1
            return

        self.attribute = ATTRIBUTES[best_attr_index]
        partitions = {}

        for i in range(len(X)):
            key = X[i][best_attr_index]
            if key not in partitions:
                partitions[key] = ([], [])
            reduced_row = X[i][:best_attr_index] + X[i][best_attr_index+1:]
            partitions[key][0].append(reduced_row)
            partitions[key][1].append(y[i])

        for attr_value, (subset_X, subset_y) in partitions.items():
            self.branches[attr_value] = DecisionTree(
                subset_X,
                subset_y,
                threshold=threshold,
                max_depth=None if max_depth is None else max_depth - 1
            )

    def entropy(self, y):
        """Entropia com numpy"""
        y = np.array(y)
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities))

    def information_gain(self, X, y, attr_index):
        total_entropy = self.entropy(y)
        partitions = {}
        for i in range(len(X)):
            key = X[i][attr_index]
            if key not in partitions:
                partitions[key] = []
            partitions[key].append(y[i])
        weighted_entropy = sum((len(subset) / len(y)) * self.entropy(subset) for subset in partitions.values())
        return total_entropy - weighted_entropy

    def choose_best_attribute(self, X, y):
        gains = [self.information_gain(X, y, i) for i in range(len(X[0]))]
        best_index = int(np.argmax(gains))
        return best_index, gains[best_index]

    def predict(self, x):  # (e.g. x = ['apple', 'green', 'circle'] -> 1 or -1)
        if self.is_leaf:
            return self.label

        attr_index = ATTRIBUTES.index(self.attribute)
        value = x[attr_index]

        if value in self.branches:
            reduced_x = x[:attr_index] + x[attr_index+1:]
            return self.branches[value].predict(reduced_x)
        else:
            return 1  


def train_decision_tree(X, y):
    return DecisionTree(X, y)
