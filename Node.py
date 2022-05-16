import pandas as pd
import numpy as np
from typing import Union, List
from sklearn.linear_model import LogisticRegression
from pandas import DataFrame
from numpy import ndarray


class Node:

    def __init__(self, x: DataFrame, y: ndarray, min_leaf=5, class_weight=None, max_depth=5, depth=0, is_root=False):
        self.x = x
        self.y = y
        self.min_leaf = min_leaf
        self.is_root = is_root
        if class_weight:
            self.class_weight = class_weight
        self.max_depth = max_depth
        self.depth = depth
        self.right_node: Union[Node, None] = None
        self.left_node: Union[Node, None] = None
        self.model = LogisticRegression()
        self.count_classes = self.get_count_of_classes(self.y)
        self.criteria = "gini"
        self.num_samples = x.shape[0]
        self.gini = Node.gini_impurity(*self.count_classes)

    def grow_tree(self):
        if self.is_root:
            pass

    def find_best_split(self, var_idx):
        pass

    def get_gini_gain(self, lhs: List[int], rhs: List[int]):
        left_values = self.y[lhs]
        right_values = self.y[rhs]
        total = (len(left_values) + len(right_values))
        p_left = len(left_values) / total
        p_right = len(right_values) / total
        y1_left, y2_left = self.get_count_of_classes(left_values)
        y1_right, y2_right = self.get_count_of_classes(right_values)
        left_impurity = Node.gini_impurity(y1_left, y2_left)
        right_impurity = Node.gini_impurity(y1_right, y2_right)
        return self.gini - (p_left*left_impurity + p_right*right_impurity)

    def is_leaf(self) -> bool:
        return True if (not self.left_node and not self.right_node) else False

    def predict(self, x):
        pass

    def predict_row(self, xi):
        pass

    def get_count_of_classes(self, y: ndarray):
        return tuple(y[y == class_count].count() for class_count in np.unique(y))

    @staticmethod
    def gini_impurity(y1_count, y2_count) -> float:
        p_y1 = y1_count / (y1_count + y2_count)
        p_y2 = y2_count / (y1_count + y2_count)
        return 1 - (p_y1 ** 2 + p_y2 ** 2)


if __name__ == '__main__':
    pass