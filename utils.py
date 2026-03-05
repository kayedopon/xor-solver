import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_csv(filepath="data.csv"):
    df = pd.read_csv(filepath)
    X = np.array(df.iloc[:, :2])
    y = np.array(df.iloc[:, 2])
    return X, y

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
    