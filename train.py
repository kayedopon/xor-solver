import numpy as np

from model import MLP
from loss import BinaryCrossEntropy
from optim import SGD
from utils import load_csv, plot_results


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
    return train_loss, train_acc * 100

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
    return test_loss, test_acc * 100

def train(model, X_train, X_test, y_train, y_test, loss_fn, optim, epochs):
    res = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    for epoch in range(epochs):

        train_loss, train_acc = train_step(model, X_train, y_train, loss_fn, optim)
        test_loss, test_acc = test_step(model, X_test, y_test, loss_fn)

        print(f"Epoch: {epoch + 1} | Train_loss: {train_loss:.4f} | Train_acc: {train_acc:.1f}% | Test_loss: {test_loss:.4f} | Test_acc: {test_acc:.1f}%")

        res["train_loss"].append(train_loss)
        res["train_acc"].append(train_acc)
        res["test_loss"].append(test_loss)
        res["test_acc"].append(test_acc)

    return res

def main():
    np.random.seed(42)
    X, y = load_csv()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # initialize hyperparameters
    EPOCHS = 4
    in_features = 2
    out_features = 1
    hidden_units = 10

    # instantiate MLP
    model = MLP(in_features, hidden_units, out_features)

    # initialize loss function and optimizer
    loss_fn = BinaryCrossEntropy()
    optim = SGD(model.parameters(), lr=0.1)

    res = train(model, X_train, X_test, y_train, y_test, loss_fn, optim, EPOCHS)
    plot_results(res)

if __name__ == "__main__":
    main()