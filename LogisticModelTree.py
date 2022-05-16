from Node import Node
from typing import Union


class LogisticModelTree:

    def __init__(self):
        self.dtree: Union[Node,None] = None

    def fit(self, X, y, min_leaf=5, class_weight=None, max_depth=5):
        self.dtree = Node(X, y, min_leaf, class_weight, max_depth=max_depth)
        self.dtree.grow_tree()
        return self

    def predict(self, X):
        return self.dtree.predict(X)

