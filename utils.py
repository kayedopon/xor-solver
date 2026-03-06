import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model import MLP


def load_csv(filepath="data.csv"):
    df = pd.read_csv(filepath)
    X = np.array(df.iloc[:, :2])
    y = np.array(df.iloc[:, 2])
    return X, y

def is_float_regex(value):
    pattern = r'^-?\d+(\.\d+)?$'
    return bool(re.match(pattern, value))

def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42):
    np.random.seed(random_state)

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

def plot_results(results):
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(results["train_loss"], label="train")
    ax[0].plot(results["test_loss"], label="test")
    ax[0].set_title("Train vs test loss")
    ax[0].legend()

    ax[1].plot(results["train_acc"], label="train")
    ax[1].plot(results["test_acc"], label="test")
    ax[1].set_title("Train vs test acc")
    ax[1].legend()

    plt.show()

def save_model(model, path="params.txt"):
    with open(path, "w") as f:
        for name, param in model.named_parameters():
            f.write(name + '\n')
            arr = param.value
            for row in arr:
                if arr.ndim == 1:
                    f.write(str(row) + "\n")
                else:
                    f.write(" ".join(map(str, row)) + '\n')
            
def load_model(model, path="params.txt"):
    with open(path, "r") as f:
        buffer = []
        label = None
        params = {}

        for line in f:
            line = line.strip('\n').split(' ')
            if line == "":
                continue

            if not is_float_regex(line[0]):
                if label is not None:
                    params[label] = np.array(buffer)
                label = line[0]
                buffer = []
            else:
                buffer.append(list(map(np.float64, line)))
        
        if label is not None:
            params[label] = np.array(buffer)

        for name, param in model.named_parameters():
            param.value = params[name]