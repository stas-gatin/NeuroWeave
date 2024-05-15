from .model import Model
from weave import Tensor, rand
from .activations import Softmax


class L1Loss(Model):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        value = abs(x - y)
        out = (value.sum(axis=1) / y.shape[1]).mean()
        return out

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        return self.forward(x, y)

    def __repr__(self) -> str:
        return 'L1Loss()'


class L2Loss(Model):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        value = (x - y) ** 2
        out = (value.sum(axis=1) / y.shape[1]).mean()
        return out

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        return self.forward(x, y)

    def __repr__(self) -> str:
        return 'L2Loss()'


class CrossEntropyLoss(Model):
    def __init__(self):
        super().__init__()
        self._softmax = Softmax()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        soft = self._softmax(x)
        out = -(((y * soft.log()) / x.shape[0]).sum())
        return out

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        return self.forward(x, y)

    def __repr__(self) -> str:
        return 'CrossEntropyLoss()'
