import pandas as pd
import numpy as np
from typing import Union, List
from sklearn.linear_model import LogisticRegression
from pandas import DataFrame
from numpy import ndarray
from dataclasses import dataclass


@dataclass
class BestSplit:
    best_gini: int
    left_index: ndarray
    right_index: ndarray
    split_value: np.float
    feature: int
    all_leaves: bool = False


class Node:

    def __init__(self, x: DataFrame, y: ndarray, min_leaf=5, class_weight=None, max_depth=5, depth=0):
        self.x = x
        self.y = y
        self.min_leaf = min_leaf
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
        self.pred: Union[bool, int] = True
        self.fit_model(self.x, self.y)
        self.best_split: Union[BestSplit, None] = None

    def grow_tree(self):
        best_split = BestSplit(
            best_gini=-1,
            left_index=None,
            right_index=None,
            split_value=0,
            feature=-1
        )
        for feature in range(self.x.shape[1]):
            feature_best_split = self.find_best_split(feature)
            if feature_best_split.best_gini > best_split.best_gini:
                best_split = feature_best_split

        if best_split.all_leaves or (best_split.right_index is None) or best_split.left_index is None or self.depth+1 > self.max_depth:
            return
        try:
            right_x = self.x[best_split.right_index]
            right_y = self.y[best_split.right_index].reset_index(drop=True)
            left_x = self.x[best_split.left_index]
            left_y = self.y[best_split.left_index].reset_index(drop=True)
        except Exception as e:
            print(e)
            return

        self.best_split = best_split
        if len(right_x) > self.min_leaf and len(left_x) > self.min_leaf:
            self.right_node = Node.init_node(right_x, right_y, self.depth + 1)
            self.left_node = Node.init_node(left_x, left_y, self.depth + 1)
            self.right_node.grow_tree()
            self.left_node.grow_tree()
        elif len(right_x) <= self.min_leaf < len(left_x):
            self.left_node = Node.init_node(left_x, left_y, self.depth + 1)
            self.left_node.grow_tree()
        elif len(right_x) > self.min_leaf >= len(left_x):
            self.right_node = Node.init_node(right_x, right_y, self.depth + 1)
            self.right_node.grow_tree()
        else:
            return

    def find_best_split(self, var_idx):
        best_split = BestSplit(
            best_gini=0,
            left_index=None,
            right_index=None,
            split_value=0,
            feature=var_idx
        )
        is_min_leaves = 0
        values = self.x[:, var_idx]
        for value in values:
            left_index = np.where(self.x[:, var_idx] <= value)[0]
            right_index = np.delete(np.arange(0, self.num_samples), left_index)
            if not self.is_valid(left_index, right_index):
                is_min_leaves += 1
                continue
            gini_index = self.get_gini_gain(left_index, right_index)
            if gini_index > best_split.best_gini:
                best_split.best_gini = gini_index
                best_split.split_value = value
                best_split.right_index = right_index
                best_split.left_index = left_index

        if is_min_leaves == len(values):
            best_split.all_leaves = True
        return best_split

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
        return self.gini - (p_left * left_impurity + p_right * right_impurity)

    def is_leaf(self) -> bool:
        return True if (not self.left_node and not self.right_node) else False

    def predict(self, x):
        pred = []
        for xi in x:
            pred.append(self.predict_row(xi))
        return pred

    def predict_row(self, xi):
        root = self
        while not root.is_leaf():
            if xi[root.best_split.feature] <= root.best_split.split_value:
                if not root.left_node:
                    break
                root = root.left_node
            else:
                if not root.right_node:
                    break
                root = root.right_node
        return [self.pred] if not self.model else self.model.predict(xi.reshape(1, -1))

    def get_count_of_classes(self, y: ndarray):
        res = []
        for label in np.unique(y):
            count_label = y[y == label].count()
            res.append(count_label)
        if len(res) == 1:
            res.append(0)
        return res

    def is_valid(self, l, r):
        if not all([len(r) > self.min_leaf,
                    len(l) >= self.min_leaf]):
            return False
        return True

    def fit_model(self, x, y):
        labels = np.unique(y)
        if len(labels) == 1:
            self.pred = labels[0]
            self.model = None
            return
        self.model.fit(x, y)

    @staticmethod
    def init_node(x, y, depth):
        return Node(x, y, depth=depth+1)

    @staticmethod
    def gini_impurity(y1_count, y2_count) -> float:
        p_y1 = y1_count / (y1_count + y2_count)
        p_y2 = y2_count / (y1_count + y2_count)
        return 1 - (p_y1 ** 2 + p_y2 ** 2)
