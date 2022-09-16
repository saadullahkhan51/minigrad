import numpy as np
# from scalar import Scalar
from tensor import Tensor

a0 = Tensor([[0.5, 1], [1, 0.5]])
w0 = Tensor(np.ones((3, 2)))
b0 = Tensor(np.random.randn(2))
z1 = w0 @ a0
a1 = z1 + b0
print(z1.shape)
# a1 = Tensor.tanh(z1)
# print(a1.grad)
# print(z1.grad)
# a1.backwards()
# print("after backprop")
# print(a1.grad)
# print(z1.grad)
# print(w0.grad)
# print(a0.grad)

# w1 = Tensor([[0.5, 0.5, 0.5]])
# a1 = w0 @ a0
# #print(a1.data)
# y = w1 @ a1
# y.backwards()
# print(y.grad, w1.grad, a1.grad)


# h = 0.01

# a = Scalar(2)
# b = Scalar(5)
# c = a + b
# d = a * b
# out = c * d
# l1 = out.data
# print(out.grad, d.grad, c.grad, b.grad, a.grad)
# out.backwards()
# print(out.grad, d.grad, c.grad, b.grad, a.grad)
# a = Scalar(2)
# b = Scalar(5)
# c = a + b
# d = a * b
# out = c * d
# l2 = out.data

# print((l2-l1)/h)
# (f(x+h)-f(x))/h

# a = [2, 3, 4, 5]
# b = [1, 2, 3, 4]
# c = a + b
# dcdb = [1, 1, 1, 1]
# dcda = [1, 1, 1, 1]