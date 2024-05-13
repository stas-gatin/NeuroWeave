from weave import Tensor
from model import Model


class ReLU(Model):
    """
    Applies the Rectified Linear Unit (ReLU) activation function to the input. It constrains them to the range
    [0, .. math::\infty)
    """
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
        pass

    def __call__(self, x: Tensor) -> Tensor:
        pass

    def __repr__(self) -> str:
        pass
