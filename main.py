import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from Node import Node
from LogisticModelTree import LogisticModelTree
from sklearn.model_selection import KFold
import json
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

CONF = {
    '1': {"min_leaf": 3, "max_depth": 3},
    '2': {"min_leaf": 5, "max_depth": 5},
    '3': {"min_leaf": 10, "max_depth": 10}
}


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


def plot_roc_curve_per_conf(roc, auc, conf):
    mean_fpr = np.mean([i[0] for i in roc[conf]], axis=0)
    mean_tpr = np.mean([i[1] for i in roc[conf]], axis=0)
    mean_auc = np.mean(auc[conf])
    plt.plot(mean_fpr, mean_tpr, label=f"ROC Curve(area={round(mean_auc, 5)})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.title(f"Roc Auc Curve for - max_depth={CONF[conf]['max_depth']} min_leaves={CONF[conf]['min_leaf']}")
    plt.legend(loc=4)
    plt.show()


def show_roc_curve(path_to_results):
    with open(path_to_results, 'r') as f:
        roc = json.loads(f.read())
    for conf in roc['roc']:
        plot_roc_curve_per_conf(roc['roc'], roc['auc'], conf)


# bonus
def calculate_class_weights(y):
    pass


def get_pie_chart(df: DataFrame):
    true = df[df["target"] == 1].shape[0]
    false = df[df["target"] == 0].shape[0]
    plt.pie([true, false], autopct=lambda x: f"{round(x, 2)}%")
    plt.legend(["True", "False"])
    font1 = {'family': 'serif', 'color': 'black', 'size': 20}
    plt.title("Target Distribution", fontdict=font1)
    plt.show()


def get_feature_hist(df: DataFrame):
    font1 = {'family': 'serif', 'color': 'black', 'size': 15}
    for col in df:
        plt.hist(df[col])
        plt.title(col, fontdict=font1)
        plt.show()


def boxplot(df: DataFrame):
    for column in df:
        plt.figure()
        df.boxplot([column], grid=False, rot=45, fontsize=15)


def get_null_values(df: DataFrame):
    print(df.isna().sum())


def get_numeric_stats(df: DataFrame):
    print(df.describe(include='all'))


def traverse_tree(root: Node):
    if root:
        traverse_tree(root.left_node)
        print(root.best_split)
        traverse_tree(root.right_node)


def cross_validation(x, y):
    conf_2 = {"min_leaf": 3, "max_depth": 3}
    conf_1 = {"min_leaf": 5, "max_depth": 5}
    conf_3 = {"min_leaf": 10, "max_depth": 10}
    confs = [conf_2, conf_1, conf_3]
    roc_auc = dict()
    auc_dic = dict()
    for i, conf in tqdm(enumerate(confs)):
        roc_auc[i + 1] = []
        auc_dic[i + 1] = []
        clf = LogisticModelTree()
        kf = KFold(n_splits=10, shuffle=True)
        folds = kf.split(x, y)
        for fold in tqdm(folds):
            x_train = x[fold[0]]
            y_train = y[fold[0]]
            x_test = x[fold[1]]
            y_test = y[fold[1]]
            clf.fit(x_train, y_train.reset_index(drop=True), min_leaf=conf["min_leaf"], max_depth=conf.get("max_depth"))
            y_pred = clf.predict(x_test)
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            auc_dic[i + 1].append(auc(fpr, tpr))
            roc_auc[i + 1].append((fpr.tolist(), tpr.tolist()))
    return {
        "roc": roc_auc,
        "auc": auc_dic
    }