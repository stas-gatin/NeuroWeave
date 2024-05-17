from weave import Tensor


class SGD:
    def __init__(self, params: list, lr: float, momentum: float = 0.9):
        self._params = params
        self._lr = lr
        self.momentum = momentum
        self._mom_v = [0] * len(params)

    def step(self):
        for i, param in enumerate(self._params):
            param -= self._lr * param.grad
            # self._mom_v[i] = self.momentum * self._mom_v[i] + (1 - self.momentum) * param.grad
            # param -= self._lr * self._mom_v[i]

    def zero_grad(self):
        for param in self._params:
            param.grad = 0
