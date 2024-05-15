import math
import numpy as np
from weave import Tensor, diag, outer, vstack
from model import Model


class ReLU(Model):
    def __init__(self):
        super().__init__()

    def forward(self, x: "Tensor") -> Tensor:
        out = Tensor(abs((x > 0) * x), _children=(x,), _op='relu', use_grad=x.grad_enabled, device=self.device)

        def _backward():
            x.grad += (out > 0) * out.grad

        if x.grad_enabled:
            out._backward = _backward
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return f"ReLU()"


class Sigmoid(Model):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(data=(1 / (1 + (-x).exp())), _children=(x,), _op='sigmoid', use_grad=x.grad_enabled,
                     device=self.device)

        def _backward():
            x.grad += ((-x).exp() / ((1 + (-x).exp()) ** 2)) * out.grad

        if x.grad_enabled:
            out._backward = _backward
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return f'Sigmoid()'


class Tanh(Model):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(data=((x.exp() - (-x).exp()) / (x.exp() + (-x).exp())), _children=(x,), _op='tanh',
                     use_grad=x.grad_enabled, device=self.device)

        def _backward():
            x.grad += (1 - ((x.exp() - (-x).exp()) / (x.exp() + (-x).exp())) ** 2) * out.grad

        if x.grad_enabled:
            out._backward = _backward
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return 'Tanh()'


class GeLU(Model):
    def __init__(self):
        super().__init__()
        self.tanh = Tanh()

    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(data=(0.5 * x * (1 + self.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))), _children=(x,),
                     _op='gelu', use_grad=x.grad_enabled, device=self.device)

        def _backward():
            # The derivative for the GeLU function is quite complicated, and I'm still debating whether implementing it
            # is even worth it, considering the heavy computational cost is going to have.
            x._grad_enabled = False
            a = self.tanh(math.sqrt(2 / math.pi) * (0.044715 * x ** 3 + x))
            b = 0.5 * math.sqrt(2) * x * (1 - a ** 2) * (0.134145 * x ** 2 + 1)
            x.grad += ((b / math.sqrt(math.pi)) + (0.5 * a) + 0.5) * out.grad
            x._grad_enabled = True

        if x.grad_enabled:
            out._backward = _backward
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return 'GeLU()'


class LeakyReLU(Model):
    def __init__(self, down_slope: float = 0.01):
        super().__init__()
        self.down_slope = down_slope

    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(data=(abs((x > 0) * x) + self.down_slope * abs((x < 0) * x)), _children=(x,), _op='leaky_relu',
                     use_grad=x.grad_enabled, device=self.device)

        def _backward():
            x.grad += np.where(x > 0, 1, self.down_slope) * out.grad

        if x.grad_enabled:
            out._backward = _backward
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return 'LeakyReLU()'


class SiLU(Model):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(data=(x / (1 + (-x).exp())), _children=(x,), _op='silu', use_grad=x.grad_enabled,
                     device=self.device)

        def _backward():
            x.grad += ((1 + (-x).exp() + x * (-x).exp()) / (1 + (-x).exp()) ** 2) * out.grad

        if x.grad_enabled:
            out._backward = _backward
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return 'SiLU()'


class Gaussian(Model):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(data=(-(x ** 2)).exp(), _children=(x,), _op='gaussian', use_grad=x.grad_enabled,
                     device=self.device)

        def _backward():
            x.grad += -2 * x * (-(x ** 2)).exp() * out.grad

        if x.grad_enabled:
            out._backward = _backward
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return 'Gaussian()'


class Softmax(Model):
    def __init__(self, dim: int = None):
        super().__init__()
        self.dim = 1 if dim is None else dim

    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(data=(x.exp() / x.exp().sum(axis=self.dim).unsqueeze().T), _children=(x,), _op='softmax',
                     use_grad=x.grad_enabled, device=self.device)

        def _backward():
            out._grad_enabled = False
            if len(out.shape) > 1:
                derivative = []
                for i in range(out.shape[0]):
                    d = out.grad @ (diag(out[i].squeeze()) - outer(out[i], out[i]))
                    derivative.append(d[i])
                x.grad += vstack(tuple(derivative))
            else:
                x.grad += out.grad @ (diag(out.squeeze()) - outer(out, out))
            out._grad_enabled = True

        if x.grad_enabled:
            out._backward = _backward
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return 'Softmax()'
