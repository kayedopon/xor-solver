import numpy as np
import pandas as pd

from model import MLP
from loss import BinaryCrossEntropy
from optim import SGD


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

def train_step(model, X_train, y_train, loss_fn, optim):
    train_loss = 0.0
    accuracy = 0
    n = len(X_train)

    for i in range(n):
        xi = X_train[i]
        yi = y_train[i]

        xi = np.asarray(xi, np.float64).reshape(1, -1)
        yi = np.asarray(yi, np.float64).reshape(1, 1)

        prob = model.forward(xi)
        loss = loss_fn.forward(prob, yi)

        optim.zero_grad()

        dl_dp = loss_fn.backward()
        model.backward(dl_dp)

        optim.step()

        train_loss += float(loss)
        pred = (prob > 0.5).astype(np.float64)
        accuracy += pred.item() == yi.item()
    
    train_loss /= n
    train_acc = accuracy / n
    return train_loss, train_acc

def test_step(model, X_test, y_test, loss_fn):
    test_loss = 0.0
    accuracy = 0
    n = len(X_test)

    for i in range(n):
        xi = X_test[i]
        yi = y_test[i]

        xi = np.asarray(xi, np.float64).reshape(1, -1)
        yi = np.asarray(yi, np.float64).reshape(1, 1)

        prob = model.forward(xi)
        loss = loss_fn.forward(prob, yi)

        test_loss += float(loss)
        pred = (prob > 0.5).astype(np.float64)
        accuracy += pred.item() == yi.item()
    
    test_loss /= n
    test_acc = accuracy / n
    return test_loss, test_acc

def train(model, X_train, X_test, y_train, y_test, loss_fn, optim, epochs):
    for epoch in range(epochs):

        train_loss, train_acc = train_step(model, X_train, y_train, loss_fn, optim)
        test_loss, test_acc = test_step(model, X_test, y_test, loss_fn)
        print(epoch)
        # to be finished

def main():
    X, y = load_csv()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # initialize hyperparameters
    EPOCHS = 2
    in_features = 2
    out_features = 1
    hidden_units = 4

    # instantiate MLP
    model = MLP(in_features, hidden_units, out_features)

    # initialize loss function and optimizer
    loss_fn = BinaryCrossEntropy()
    optim = SGD(model.parameters(), lr=0.1)

    res = train(model, X_train, X_test, y_train, y_test, loss_fn, optim, EPOCHS)

if __name__ == "__main__":
    main()