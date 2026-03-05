import numpy as np

from layers import Linear, Tanh, Sequential, Sigmoid
from base import Parameter, Module

class MLP(Module):
    """
    Multi Layer Perceptron with three layers (input, hidden, output) that 
    return probabilities rather than logits due to presence of sigmoid.

    Args:
        in_features (int default=2): Number of features in the data to be used.
        hidden_units (int default=4): Number of neurons in each hidden layer.
        out_features (int default=1): Number of values staged for output.
    """
    def __init__(self, in_features:int=2, hidden_units:int=4, out_features:int=1):
        super().__init__()
        self.mlp = Sequential(
            Linear(in_features, hidden_units),
            Tanh(),
            Linear(hidden_units, out_features),
            Sigmoid(),
        )
    
    def forward(self, x):
        return self.mlp.forward(x)

    def backward(self, dout):
        return self.mlp.backward(dout)