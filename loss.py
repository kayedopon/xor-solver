import numpy as np


class BinaryCrossEntropy:
    """
    The implementaion of BCE loss function.
    Probabilities are expected as input, not logits.
    """
    def __init__(self):
        self.y = None
        self.p = None

    def forward(self, p, y):
        # p is (0, 1), y is in {0, 1}
        eps = 1e-7
        self.y = y
        # use clipping to keep p in range (eps, 1 - eps), so log(0) won't happen
        self.p = np.clip(p, eps, 1 - eps) 
        loss = -(y * np.log(self.p) + (1 - y) * np.log(1 - self.p))
        return np.mean(loss)
    
    def backward(self):
        # dL/dp
        n = self.y.shape[0]
        return ((self.p - self.y) / (self.p * (1 - self.p))) / n