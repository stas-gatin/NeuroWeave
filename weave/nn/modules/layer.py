from weave import Tensor
from model import Model
from weave import rand


class LayerDense(Model):
    def __init__(self, in_neurons: int, out_neurons: int):
        self._weights = rand((in_neurons, out_neurons), use_grad=True, device='cpu')  # tensor made of weights
