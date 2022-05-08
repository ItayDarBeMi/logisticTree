
class Node:

    def __init__(self, x, y, min_leaf=5, class_weight=None, max_depth=5, depth=0):
        pass

    def grow_tree(self):
        pass

    def find_best_split(self, var_idx):
        pass

    def get_gini_gain(self, lhs, rhs):
        pass

    def is_leaf(self):
        pass

    def predict(self, x):
        pass

    def predict_row(self, xi):
        pass

    @staticmethod
    def gini_impurity(y1_count, y2_count):
        pass
