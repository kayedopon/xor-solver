import numpy as np


class Parameter:
    def __init__(self, value):
        self.value = value.astype(np.float64)
        self.grad = np.zeros_like(value)
    
    def zero_grad(self):
        self.grad[...] = 0.0
        

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value

        object.__setattr__(self, name, value)

    def parameters(self):
        for p in self._parameters.values():
            yield p

        for m in self._modules.values():
            yield from m.parameters()

    def named_parameters(self, prefix=""):
        for name, p in self._parameters.items():
            yield prefix + name, p

        for name, m in self._modules.items():
            yield from m.named_parameters(prefix + name + ".")