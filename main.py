import pandas as pd
from sklearn.model_selection import train_test_split


def analyze_data(data):
    pass


def preprocess(data):
    pass


def show_roc_curve(y_true, preds):
    pass


# bonus
def calculate_class_weights(y):
    pass


if __name__ == "__main__":
    data = pd.read_csv('heart.csv')
    x, y = preprocess(data)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=11)
    # weights = calculate_class_weights(y)

    # implement here the experiments for task 4
