from weave import Tensor
from model import Model
from weave import rand


class LayerDense(Model):
    def __init__(self, in_neurons: int, out_neurons: int, bias: bool = True):
        super().__init__()
        self._weights = rand((in_neurons, out_neurons), use_grad=True, device='cpu')  # tensor made of weights
        if bias:
            self._bias = rand((1, out_neurons), use_grad=True, device='cpu')

    def forward(self, inputs: Tensor) -> Tensor:
        if hasattr(self, '_bias'):
            return inputs @ self._weights + self._bias
        return inputs @ self._weights

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

    def __repr__(self):
        return (f'LayerDense(in_neurons={self._weights.shape[0]}, out_neurons={self._weights.shape[1]}, '
                f'bias={hasattr(self, '_bias')})')
