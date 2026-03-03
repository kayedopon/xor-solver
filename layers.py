import numpy as np


class Parameter:
    def __init__(self, value):
        self.value = value.astype(np.float64)
        self.grad = np.zeros_like(value)
    
    def zero_grad(self):
        self.grad[...] = 0.0

class Linear:
    """
    Linear class is an implementaion of linear layer that solve the following
    equation: y = xw + b.
    Args: 
        in_features (int): number of features for input.
        out_features (int): number of features staged for output
    """
    def __init__(self, in_dim, out_dim):
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
    
    def parameters(self):
        return [self.W, self.b]


class Sigmoid:
    """
    The implementation of Sigmoid activation function: `1/(1+e^-x)`
    """
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        return dout * (self.out * (1 - self.out))
    
    def parameters(self):
        return []
    

class Tanh():
    """
    The implementation of Tanh activation function: `(e^x - e^-x)/(e^x + e^-x)`
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
    
    def parameters(self):
        return []


class Sequential:
    """
    Sequential layer allows you to run layers provided consequently.
    """
    def __init__(self, *args):
        self.sequence = args

    def forward(self, x):
        for l in self.sequence:
            x = l.forward(x)
        return x
    
    def backward(self, dout):
        for l in reversed(self.sequence):
            dout = l.backward(dout)
        return dout
    
    def parameters(self):
        parameters = []
        for l in self.sequence:
            parameters.extend(l.parameters())
        return parameters