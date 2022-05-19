from Node import Node
from typing import Union


class LogisticModelTree:

    def __init__(self, min_leaf=5, class_weight=None, max_depth=5):
        self.min_leaf = min_leaf
        self.class_weight = class_weight
        self.max_depth = max_depth
        self.dtree: Union[Node, None] = None

    def fit(self, X, y):
        self.dtree = Node(X, y, min_leaf=self.min_leaf, class_weight=self.class_weight, max_depth=self.max_depth)
        self.dtree.grow_tree()
        return self

    def predict(self, X):
        return self.dtree.predict(X)
