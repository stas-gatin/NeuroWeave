from model import Model
from weave import Tensor, rand
import numpy as np


class L2(Model):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        value = (x - y) ** 2
        if len(value.shape) < 2:
            value = value.unsqueeze(0)
        elif len(value.shape) == 2:
            idxs = np.where(np.array(value.shape) == 1)[0]
            while 1 in value.shape:
                value = value.squeeze(value.shape.index(1))
        print(value)
        final = []
        for row in value:
            v = Tensor(data=0, dtype=float, use_grad=x.grad_enabled, device=self.device)
            for element in row:
                v += element
            final.append(v.data.item())
        final = Tensor()


    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        return self.forward(x, y)

    def __repr__(self) -> str:
        pass


if __name__ == '__main__':
    np.random.seed = 42
    a = Tensor(data=[[0.52212688, 0.11074398, 0.33256006, 0.74944895],
                     [0.29366877, 0.86421459, 0.14593383, 0.10099576],
                     [0.00644682, 0.02650849, 0.64224052, 0.97380977]], use_grad=True, device='cpu')
    b = Tensor(data=[[0.60643317, 0.34354128, 0.2734914, 0.087636],
                     [0.80810851, 0.77430177, 0.58968519, 0.64265502],
                     [0.25375612, 0.51153184, 0.08059557, 0.5863689]], use_grad=True, device='cpu')
    print(a.sum(0))

