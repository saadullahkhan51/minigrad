import numpy as np
class Tensor:
    def __init__(self, data, children = [], requires_grad = True):
        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
        self.children = set(children)
        if requires_grad:
            self.grad = np.zeros(self.shape)
        self.backward = lambda: None
    def __add__(self, op):
        op = op if isinstance(op, Tensor) else Tensor(op)
        res = Tensor(self.data + op.data, [self, op])
        
        def backward():
            self.grad += res.grad
            op.grad += res.grad
        res.backward = backward
        return res

    def __matmul__(self, op):
        op = op if isinstance(op, Tensor) else Tensor(op)
        res = Tensor(self.data @ op.data, [self, op])
        def backward():
            self.grad += res.grad @ op.data.T
            op.grad += self.data.T @ res.grad
        res.backward = backward
        return res
    
    def tanh(self):
        res = Tensor(np.tanh(self.data), [self])
        deriv = np.array(1. - res.data**2)
        def backward():
            self.grad += res.grad * deriv
        res.backward = backward
        return res
        
    def __repr__(self):
        return f"Tensor:(data = {self.data})"

    def backwards(self):
        order = []
        seen = set()
        def topsort(node):
            if node not in seen:
                seen.add(node)
                for child in node.children:
                    topsort(child)
                order.append(node)
        topsort(self)
        
        self.grad = np.ones(self.shape)
        for node in reversed(order):
            node.backward()
    
    def __radd__(self, op):
        return self + op
    def __neg__(self):
        return self.data * -1
    def __sub__(self, op):
        return self + (-op)
    def __rsub__(self, op):
        return op + (-self)