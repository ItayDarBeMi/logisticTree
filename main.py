import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from Node import Node
from sklearn.metrics import accuracy_score
from LogisticModelTree import LogisticModelTree
from sklearn.model_selection import cross_val_score, KFold
import json
from sklearn.metrics import roc_auc_score,auc,roc_curve
from tqdm import tqdm


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


def show_roc_curve(path_to_results):



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


def cross_validation(x,y):
    conf_2 = {"min_leaf":3,"max_depth":3}
    conf_1 = {"min_leaf":5,"max_depth":5}
    conf_3 = {"min_leaf":10,"max_depth":10}
    confs = [conf_2,conf_1,conf_3]
    roc_auc = dict()
    auc_dic = dict()
    for i,conf in tqdm(enumerate(confs)):
        roc_auc[i+1] = []
        auc_dic[i+1] = []
        clf = LogisticModelTree(min_leaf=conf["min_leaf"],max_depth=conf.get("max_depth"))
        kf = KFold(n_splits=10, random_state=42, shuffle=True)
        folds = kf.split(x, y)
        for fold in tqdm(folds):
            x_train = x[fold[0]]
            y_train = y[fold[0]]
            x_test = x[fold[1]]
            y_test = y[fold[1]]
            clf.fit(x_train,y_train.reset_index(drop=True))
            y_pred = clf.predict(x_test)
            fpr, tpr, _ = roc_curve(y_test,y_pred)
            auc_dic[i+1].append(auc(fpr,tpr))
            roc_auc[i+1].append((fpr.tolist(),tpr.tolist()))
    return {
        "roc": roc_auc,
        "auc": auc_dic
    }



# def auc_graph(scores):
#     from sklearn.metrics import roc_auc_score
#     # auc scores
#     auc_score = roc_auc_score(y_test, pred_prob1[:, 1])
#     auc_score2 = roc_auc_score(y_test, pred_prob2[:, 1])
#
#     print(auc_score1, auc_score2)
#     from sklearn.metrics import roc_curve

if __name__ == "__main__":
    data = pd.read_csv('heart.csv')
    # analyze_data(data)
    x, y = preprocess(data)
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # root = Node(X_train, y_train.reset_index(drop=True))
    # root.grow_tree()
    # y_pred = root.predict(X_test)
    # print(accuracy_score(y_test,y_pred))
    # cross_validation(x,y)
    # cv = KFold(n_splits=10,random_state=42,shuffle=True)
    # z = cv.split(x,y)
    # print(z)
    ret = cross_validation(x,y)
    try:
        with open("ruc_auc.json",'w') as results:
            json.dump(ret,results)
    except Exception as e:
        print(e)




    # implement here the experiments for task 4
