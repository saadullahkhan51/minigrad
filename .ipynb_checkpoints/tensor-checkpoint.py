import numpy as np


class Tensor:
    def __init__(self, data, children=[], requires_grad=True):
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
    
    # "In general, if we have an N×N filter, a W×H image, and stride S there are (W-N) //S + 1 places to apply the filter ..."
    def conv1d(self, kernel, stride=1):
        out_shape = (self.shape[0] - kernel.shape[0]) // stride + 1
        output = Tensor(np.zeros(out_shape), [self, kernel])

        for i in range(0, self.shape[0] - kernel.shape[0] + 1, stride):
            output.data[i] = (self.data[i:i+kernel.shape[0]] * kernel.data).sum()

        def backward():
            self.grad += kernel.data * output.grad
            kernel.grad += self.data[:output.shape[0]] * output.grad

            self.grad[kernel.shape[0]:] = 0
        output.backward = backward

        return output
    
    def conv2d(self, kernel, stride=1):
        out_shape = ((self.shape[0] - kernel.shape[0]) // stride + 1, (self.shape[1] - kernel.shape[1]) // stride + 1)
        output = Tensor(np.zeros(out_shape))
        # fm.data[0][0] = (S3.data[0][0:2] * k3.data[0]).sum() + (S3.data[1][0:2] * k3.data[1]).sum()
        # fm.data[0][1] = (S3.data[0][1:3] * k3.data[0]).sum() + (S3.data[1][1:3] * k3.data[1]).sum()
        # fm.data[1][0] = (S3.data[1][0:2] * k3.data[0]).sum() + (S3.data[2][0:2] * k3.data[1]).sum()
        # fm.data[1][1] = (S3.data[1][1:3] * k3.data[0]).sum() + (S3.data[2][1:3] * k3.data[1]).sum()

        #fm.data[i][j] = (S3.data[i][j:j+k3.shape[0]] * k3.data[0]).sum() + (S3.data[i+1][j:j+k3.shape[0]] * k3.data[1]).sum()

        # the indexing seems to be working, hardcoded for a 2x2 filter/kernel
        for i in range(0, out_shape[1], stride):
            for j in range(0, out_shape[0], stride):
                output.data[i][j] = (self.data[i][j:j+kernel.shape[0]] * kernel.data[0]).sum() + (self.data[i+1][j:j+kernel.shape[0]] * kernel.data[1]).sum()
        return output
                
        # TODO
#         def backward():
#             self.grad += 1
#             kernel.grad += 
