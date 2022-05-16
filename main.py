import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from Node import Node


def analyze_data(data):
    get_pie_chart(data)
    get_null_values(data)
    get_numeric_stats(data)
    get_feature_hist(data)


def preprocess(data):
    standard_scaler = StandardScaler()
    data = data.fillna({col: np.mean(data[col]) for col in data})
    y = data["target"]
    X = data.drop("target", axis=1)
    scaled_data = standard_scaler.fit_transform(X)
    return scaled_data, y


def show_roc_curve(y_true, preds):
    pass


# bonus
def calculate_class_weights(y):
    pass


def get_pie_chart(df: DataFrame):
    true = df[df["target"] == 1].shape[0]
    false = df[df["target"] == 0].shape[0]
    plt.pie([true, false], autopct=lambda x: f"{round(x, 2)}%")
    plt.legend(["1", "0"])
    plt.show()


def get_null_values(df: DataFrame):
    print(df.isna().sum())


def get_numeric_stats(df: DataFrame):
    print(df.describe(include='all'))


def get_feature_hist(df: DataFrame):
    for col in df:
        plt.hist(df[col])
        plt.show()


def traverse_tree(root: Node):
    if root:
        traverse_tree(root.left_node)
        print(root.best_split)
        traverse_tree(root.right_node)


if __name__ == "__main__":
    data = pd.read_csv('heart.csv')
    # analyze_data(data)
    x, y = preprocess(data)
    root = Node(x, y)
    root.grow_tree()
    traverse_tree(root)

    # for index in range(x.shape[1]):
    #     print(n.find_best_split(index).right_index)
    # idx_left = np.where(x[:, 0] <= 0.00234)[0]
    # idx_right = np.delete(np.arange(0, len(x)), idx_left)
    # print(n.get_gini_gain(idx_left,idx_right))

    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=11)
    # weights = calculate_class_weights(y)

    # implement here the experiments for task 4
