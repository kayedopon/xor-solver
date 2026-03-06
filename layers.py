import numpy as np

from base import Parameter, Module

class Linear(Module):
    """
    Linear class is an implementaion of linear layer that solve the following equation: y = xw + b.

    Args: 
        in_features (int): number of features for input.
        out_features (int): number of features staged for output
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = Parameter(np.random.rand(in_dim, out_dim) * (2 / np.sqrt(in_dim)))
        self.b = Parameter(np.zeros((1, out_dim)))
        self.x = None

    def forward(self, x):
        # x: (batch, in_dim)
        # W: (in_dim, out_dim)
        # b: (1, out_dim)
        self.x = x
        out = x @ self.W.value + self.b.value # the output will be of shape (batch, out_dim)
        return out
    
    def backward(self, dout):
        # dout: (batch, out_dim)
        self.W.grad += self.x.T @ dout
        self.b.grad += np.sum(dout, axis=0, keepdims=True)
        dx = dout @ self.W.value.T
        return dx


class Sigmoid():
    """
    The implementation of Sigmoid activation function.

    Computes: 
        `1/(1+e^-x)`
    """
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        return dout * (self.out * (1 - self.out))
        

class Tanh():
    """
    The implementation of Tanh activation function.

    Computes:
        `(e^x - e^-x)/(e^x + e^-x)`
    """
    def __init__(self):
        self.out = None
        self.x = None
    
    def forward(self, x):
        self.x = x
        self.out = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return self.out
    
    def backward(self, dout):
        # the derivative used here was manually calculated and is the same as the known 1 - tanh(x)**2
        return dout * ((4 * np.exp(2 * self.x)) / (np.exp(2 * self.x) + 1) ** 2)


class Sequential(Module):
    """
    Sequential layer allows you to run layers provided consequently.
    """
    def __init__(self, *args):
        super().__init__()
        self.layers = []

        for i, layer in enumerate(args):
            setattr(self, str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x
    
    def backward(self, dout):
        for l in reversed(self.layers):
            dout = l.backward(dout)
        return dout