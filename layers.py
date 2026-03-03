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
        self.b.grad += np.sum(dout, axis=0, keepdim=True)
        dx = dout @ self.W.T
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


class BinaryCrossEntropy:
    def __init__(self):
        self.y = None
        self.p = None

    def forward(self, p, y):
        # p is (0, 1), y is in {0, 1}
        # Sigmoid activation function is supposed to be used bofore BCE loss
        eps = 1e-7
        self.y = y
        # use clipping to restrict p to (eps, 1 - eps), so log(0) won't happen
        self.p = np.clip(p, eps, 1 - eps) 
        loss = -(y * np.log(self.p) + (1 - y) * np.log(1 - self.p))
        return np.mean(loss)
    
    def backward(self):
        # dL/dp

        n = self.y.shape[0]
        return ((self.p - self.y) / (self.p * (1 - self.p))) / n


class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = [np.zeros_like(p.value) for p in self.params]
        self.v = [np.zeros_like(p.value) for p in self.params]
        
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            if self.weight_decay != 0:
                g = g+ self.weight_decay * p.value
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) *  p.grad
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * p.grad ** 2

            mh = self.m[i] / (1 - self.betas[0] ** self.t)
            vh = self.v[i] / (1 - self.betas[1] ** self.t)

            p.value = p.value - self.lr * (mh / (np.sqrt(vh) + self.eps))


class Sequential:
    def __init__(self, *args):
        self.sequence = args

    def forward(self, x):
        for l in self.sequence:
            x = l.forward(x)
        return x
    
    def parameters(self):
        parameters = []
        for l in self.sequence:
            parameters.extend(l.parameters())
        return parameters


class MLP:
    def __init__(self, in_features, hidden_units, out_features):
        pass

l = Linear(2, 12)
l2 = Linear(12, 12)
l3 = Linear(12, 1)
tanh = Tanh()

s = Sequential(l, l2, l3)
x = np.array([[1, 0], [1, 0]])
print(s.forward(x))