import numpy as np
from tensor import Tensor

# should not be accesible outside class?
# add requires_grad when default is set to false

class Linear():
    def __init__(self, nins, nouts, bias=False):
        self.w = Tensor(np.random.randn(nins, nouts))
        self.bias = bias
        if self.bias:
            self.b = Tensor(np.random.randn(nouts))

    def __call__(self, inputs):
        z = inputs @ self.w
        if self.bias:
            z += self.b

        return z
