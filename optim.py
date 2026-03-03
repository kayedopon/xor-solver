import numpy as np


class Adam:
    """
    The implementaion of Adam optimizer with weight decay.
    """
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
                g = g + self.weight_decay * p.value

            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) *  g
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * g ** 2

            mh = self.m[i] / (1 - self.betas[0] ** self.t)
            vh = self.v[i] / (1 - self.betas[1] ** self.t)

            p.value = p.value - self.lr * (mh / (np.sqrt(vh) + self.eps))

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()


# I am  going to use SGD because Adam is reduntant for XOR solving
class SGD:
    """
    Implementaion of Stogastic Gradient Descend optimizer: `w = w - lr * g`.
    """
    def __init__(self, params, lr=0.1):
        self.params = params
        self.lr = lr
    
    def step(self):
        for param in self.params:
            param.value -= self.lr * param.grad
    
    def zero_grad(self):
        for param in self.params:
            param.zero_grad()