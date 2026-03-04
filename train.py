import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def load_csv(filepath="data.csv"):
    df = pd.read_csv(filepath)
    X = np.array(df.iloc[:, :2])
    y = np.array(df.iloc[:, 2])
    return X, y

def train_test_split(X, y, test_size=0.2, shuffle=True):
    n = len(X)
    train_length= int(n * (1 - test_size))

    indices = np.arange(n)
    
    if shuffle == True:
        np.random.shuffle(indices)

    train_idx = indices[:train_length]
    test_idx = indices[train_length:]

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test




def main():
    X, y = load_csv()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(len(X_train), len(X_test), len(y_train), len(y_test))

if __name__ == "__main__":
    main()