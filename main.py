import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def analyze_data(data):
    get_pie_chart(data)
    get_null_values(data)
    get_numeric_stats(data)
    get_feature_hist(data)


def preprocess(data):
    pass


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
        plt.legend()
        plt.show()


if __name__ == "__main__":
    data = pd.read_csv('heart.csv')
    analyze_data(data)
    # x, y = preprocess(data)
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=11)
    # weights = calculate_class_weights(y)

    # implement here the experiments for task 4
