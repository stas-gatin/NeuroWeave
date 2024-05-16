from typing import List
from weave import Tensor
from .model import Model
from weave import rand


class LayerDense(Model):
    def __init__(self, in_neurons: int, out_neurons: int, bias: bool = True):
        super().__init__()
        self._weights = rand((in_neurons, out_neurons), use_grad=True, device=self.device)  # tensor made of weights
        if bias:
            self._bias = rand((1, out_neurons), use_grad=True, device=self.device)

    def forward(self, inputs: Tensor) -> Tensor:
        if hasattr(self, '_bias'):
            return inputs @ self._weights + self._bias
        return inputs @ self._weights

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

    def __repr__(self):
        return (f'LayerDense(in_neurons={self._weights.shape[0]}, out_neurons={self._weights.shape[1]}, '
                f'bias={hasattr(self, '_bias')})')


class Sequential(Model):
    def __init__(self, layers: list):
        super().__init__()
        self._models = layers

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._models:
            x = layer(x)
        return x

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def __repr__(self):
        string = f'Sequential('
        for i, layer in enumerate(self._models):
            string += f'\n    ({i}): {layer}'
        if self._layers:
            string += ')'
        else:
            string += '\n)'
        return string
