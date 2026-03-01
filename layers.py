from numpy import np 


class Parameter:
    def __init__(self, value):
        self.value = value.astype(np.float64)
        self.grad = np.zeros_like(value)
    
    def zero_grad(self):
        self.grad[...] = 0.0
