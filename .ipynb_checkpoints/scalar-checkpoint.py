class Scalar:
    def __init__(self, data, children=[]):
        self.data = data
        self.grad = 0
        self.children = set(children)
        self.derivate = lambda: None
        self._op = ""

    def __repr__(self):
        return f"Scalar(data = {self.data})"

    def __add__(self, operand):
        # res = self.data + operand.data
        # res.children = [self, operand]
        # res.op = '+'
        operand if isinstance(operand, Scalar) else Scalar(operand)
        res = Scalar(self.data + operand.data, [self, operand])

        def derivate():
            self.grad += res.grad
            operand.grad += res.grad
        res.derivate = derivate

        return res

    def __radd__(self, other):  # other + self
        return self + other

    def __mul__(self, operand):
        operand if isinstance(operand, Scalar) else Scalar(operand)
        res = Scalar(self.data * operand.data, [self, operand])

        def derivate():
            self.grad += operand.data * res.grad
            operand.grad += self.data * res.grad
        res.derivate = derivate

        return res

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

        self.grad = 1
        for node in reversed(order):
            node.derivate()

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, operand):
        return self + (-operand)

    def __rsub__(self, operand):
        return operand + (-self)

    def __rmul__(self, operand):
        return self * operand

    def __radd__(self, operand):
        return self * operand

    def __truediv__(self, operand):
        return self * operand**-1

    def __rtruediv__(self, operand):
        return operand * self**-1
