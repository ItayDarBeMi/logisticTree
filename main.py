import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from Node import Node
from sklearn.metrics import accuracy_score


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
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    root = Node(X_train, y_train.reset_index(drop=True))
    root.grow_tree()
    y_pred = root.predict(X_test)
    print(accuracy_score(y_test,y_pred))


    # implement here the experiments for task 4
