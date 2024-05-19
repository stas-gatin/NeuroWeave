from weave import Tensor
from .model import Model
from weave import uniform
import numpy as np


class LayerDense(Model):
    """
    Applies a linear transformation to the incoming data according to the following formula:
    y = WX + b
    Where W are the weights of the layer, X are the inputs and b is the bias vector.

    Parameters
    -----------------------------------------------
    in_neurons: int
        Number of elements in the last dimension of the inputs.
    out_neurons: int
        Desired number of elements in the last dimension of the outputs.

    Attributes
    -----------------------------------------------
    weights : weave.Tensor
        A tensor representing the weights of the layer
    bias: weave.Tensor
        A tensor representing the bias vector of the layer. It applies over all dimensions of the inputs via
        broadcasting.

    Methods
    -----------------------------------------------
    forward(inputs: weave.Tensor)
        Applies a linear transformation to the inputs. The specific formula depends on whether bias is True or False.

    """
    def __init__(self, in_neurons: int, out_neurons: int, bias: bool = True):
        super().__init__()
        self.weights = uniform(-np.sqrt(1 / in_neurons), np.sqrt(1 / in_neurons), (in_neurons, out_neurons),
                               use_grad=True, device=self.device)
        if bias:
            self.bias = uniform(-np.sqrt(1 / in_neurons), np.sqrt(1 / in_neurons), (1, out_neurons),
                                use_grad=True, device=self.device)

    def forward(self, inputs: Tensor) -> Tensor:
        if hasattr(self, '_bias'):
            return inputs @ self.weights + self.bias
        return inputs @ self.weights

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

    def __repr__(self):
        return (f'LayerDense(in_neurons={self.weights.shape[0]}, out_neurons={self.weights.shape[1]}, '
                f'bias={hasattr(self, 'bias')})')


class Sequential(Model):
    """
    A sequential block of layers. Models are passed as a list and they will be executed in the order they where passed.

    Parameters
    -----------------------------------------------
    layers: list
            A list of Models that the Sequential block will contain.

    Methods
    -----------------------------------------------
    forward(inputs: Tensor)
        Executes the forward pass of each of the layers in the Sequential block in the order they were passed.
    """
    def __init__(self, layers: list):
        super().__init__()
        self._models = layers
        for i, layer in enumerate(layers):
            setattr(self, f'sequential_{i}', layer)

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
            string += '\n)'
        else:
            string += ')'
        return string
