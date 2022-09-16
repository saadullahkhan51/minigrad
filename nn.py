import numpy as np
from tensor import Tensor

class Linear():
    def __init__(self, nins, nouts, bias = False):
        self.W = Tensor(np.random.randn(nins, nouts))
        self.bias  = bias
        if self.bias:
            self.B = Tensor(np.random.randn(nouts))
        
    def forward(self, inputs, W: Tensor, B: Tensor):
        Z = W @ inputs
        if self.bias:
            Z += B

        return Z
