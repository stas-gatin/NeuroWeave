from model import Model
from weave import Tensor, rand
import numpy as np


class L2(Model):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        value = (x - y) ** 2
        out = (value.sum(axis=0) / y.shape[1]).mean()
        return out

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        return self.forward(x, y)

    def __repr__(self) -> str:
        pass


if __name__ == '__main__':
    loss_fn = L2()
    a = Tensor(data=[[0.52212688, 0.11074398, 0.33256006, 0.74944895],
                     [0.29366877, 0.86421459, 0.14593383, 0.10099576],
                     [0.00644682, 0.02650849, 0.64224052, 0.97380977]], use_grad=True, device='cpu')
    b = Tensor(data=[[0.60643317, 0.34354128, 0.2734914, 0.087636],
                     [0.80810851, 0.77430177, 0.58968519, 0.64265502],
                     [0.25375612, 0.51153184, 0.08059557, 0.5863689]], use_grad=True, device='cpu')
    loss = loss_fn(b, a)
    print(loss.__repr__())
