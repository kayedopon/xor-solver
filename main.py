import numpy as np

from model import MLP
from loss import BinaryCrossEntropy
from optim import SGD
from utils import load_csv, plot_results, train_test_split, save_model, load_model
from train import train


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

    # make a prediction and check if it is correct
    pred = model.predict(X_test).squeeze()
    print(f"Same type: {type(pred) == type(y_test)} | Same shape: {pred.shape == y_test.shape}")
    print(f"Predictions are the same as target: {(pred == y_test).all()}")

    save_model(model)

    # load the model's parameters into new model2 with the same architecture
    model2 = MLP(in_features, hidden_units, out_features)
    load_model(model2)
    
    # check whether the parameters loaded correctly
    print(all(np.array_equal(x.value, y.value) for x, y in zip(model.parameters(), model2.parameters())))


if __name__ == "__main__":
    main()