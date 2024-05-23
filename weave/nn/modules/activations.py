"""
MIT License

Copyright (c) 2024 NeuroWeave

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
import numpy as np
from weave import Tensor, diag, outer, vstack
from .model import Model


class ReLU(Model):
    """
    Applies the Rectified Linear Unit activation function over the inputs. The ReLU function is described as:
    ReLU(X) = max(0, X)
    Where X is the input to the ReLU function.

    Methods
    -----------------------------------------------
    forward(input: Tensor)
        Applies the ReLU function over the inputs.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: "Tensor") -> Tensor:
        out = Tensor(data=abs((x > 0) * x), _children=(x,), _op='relu', use_grad=x.grad_enabled, device=self.device)

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
    """
    Applies the Sigmoid activation function over the inputs. The Sigmoid function is described as:
    Sigmoid(X) = 1 / (1 + exp(-X))
    Where X is the input to the Sigmoid function.

    Methods
    -----------------------------------------------
    forward(input: Tensor)
        Applies the Sigmoid function over the inputs.
    """
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
    """
    Applies the Hyperbolic Tangent function over the inputs. Tanh is defined as:
    Tanh(X) = exp(X) - exp(-X) / exp(X) + exp(-X)
    Where X is the input to the Tanh function.

    Methods
    -----------------------------------------------
    forward(input: Tensor)
        Applies the Tanh function over the inputs.
    """
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


class GELU(Model):
    """
    Applies the Gaussian Error Linear Unit over the inputs. The GELU function is defined as:
    GELU(X) = 0.5 * X * (1 + erf(X / sqrt(2)))
    where X is the input to the GeLU function.

    Methods
    -----------------------------------------------
    forward(input: Tensor)
        Applies the GELU function over the inputs.
    """
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
        return 'GELU()'


class LeakyReLU(Model):
    """
    Applies a leaky ReLU activation to the inputs. This activation function is defined as:
    Leaky ReLU(X) = X if X > 0 else 0.01 * X
    Where X is the input to the leaky ReLU function.

    Parameters
    -----------------------------------------------
    down_slope: float
        The amount of the original input to be preserved if the input is bellow 0.

    Methods
    -----------------------------------------------
    forward(x: Tensor)
        Applies the leaky ReLU function over the input.
    """
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
    """
    Implementation of the SiLU activation function. This function is described as:
    SiLU(X) = X / (1 + exp(-X))
    Where X is the input to the SiLU function.

    Methods
    -----------------------------------------------
    forward(x: Tensor)
        Applies the SiLU function over the input.
    """
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
    """
    Applies the Gaussian activation function over the inputs. The Gaussian function is defined as:
    Gaussian(X) = exp(-(X ** 2))
    Where X is the input to the Gaussian function.

    Methods
    -----------------------------------------------
    forward(x: Tensor)
        Applies the Gaussian function over the input.
    """
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
    """
    Implementation of the Softmax activation function. The Softmax activation function is defined as:
    Softmax(X) = exp(X_i) / Sum(exp(X_j))_(j = 1)
    where X_i and X_j are indices to the different rows of X.

    Parameters
    -----------------------------------------------
    dim: int
        Dimension over which to apply the softmax function

    Methods
    -----------------------------------------------
    forward(x: Tensor)
        Applies the Softmax activation function over the input.
    """
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
                    d = out.grad @ (diag(out[i].squeeze(), device=self.device) - outer(out[i], out[i],
                                                                                       device=self.device))
                    if d.shape[0] == 1:
                        d = d.squeeze()
                    derivative.append(d[i])
                x.grad += vstack(tuple(derivative), device=self.device)
            else:
                x.grad += out.grad @ (diag(out.squeeze(), device=self.device) - outer(out, out, device=self.device))
            out._grad_enabled = True

        if x.grad_enabled:
            out._backward = _backward
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return 'Softmax()'
