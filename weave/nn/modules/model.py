import weave.nn
from weave import Tensor
from weave.cuda import Device
import re
import numpy as np


# We created the base class for all models as well as its metaclass to run functions in the background
class ModelMeta(type):
    def __new__(cls, name, bases: tuple, attr: dict):
        if '__init__' in attr.keys():
            original_init = attr['__init__']

            def init_wrapper(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                self._parameter_buffer()
            attr['__init__'] = init_wrapper
        return super().__new__(cls, name, bases, attr)


class Model(metaclass=ModelMeta):  # Model is a layer with layers inside (like an onion)
    def __init__(self, device: str | Device = 'cpu'):
        self.device = Device(device)

    def forward(self, *tensors) -> Tensor:  # We take args to satisfy the needs of losses forward function
        pass

    def __call__(self, *tensors) -> Tensor:
        return self.forward(*tensors)

    def _parameter_buffer(self):
        self._parameters = {}  # dictionary for tensors
        self._layers = {}  # Dictionary for layers
        self._num_parameters = 0
        for attr in dir(self):
            try:
                value = getattr(self, attr)  # in order to call an attribute by using an str
            except AttributeError:
                continue  # next for iteration
            else:
                if isinstance(value, Tensor):
                    self._parameters[attr] = value
                    self._num_parameters += value.size
                elif isinstance(value, Model):
                    self._layers[attr] = value
        for name, layer in self._layers.items():
            self._parameters.update({f'{name}': layer._parameters})  # we add a dictionary to the created one (concatenar)
            self._num_parameters += layer._num_parameters

    def data_dict(self):
        model_dict = {'weights': [*self.params()]}
        model_dict['config'] = {
            'name': self.__class__.__name__,
            'num_layers': len(self._layers),
            'num_parameters': len(model_dict),
        }
        return model_dict

    def params(self):
        return self._depth_call('get')

    # Load values from a trained model
    @classmethod
    def load(cls, values: dict, *args):
        m = cls(*args)  # classmethod is empty

        def rec_fill(obj, weights):
            obj._layers = {}
            for name, value in obj._parameters.items():
                if isinstance(value, Tensor):
                    setattr(obj, name, weights[0])
                    del weights[0]
                elif isinstance(value, dict):
                    obj._layers[name] = getattr(obj, name)
                    rec_fill(getattr(obj, name), weights)
        values: list = values['weights']
        rec_fill(m, values)
        return m

    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}('
        for name, value in self._layers.items():
            if not isinstance(value, weave.nn.Sequential):
                s += f'\n    ({name}): {value}'
            else:
                seq_string = value.__repr__()
                seq_string = re.sub(r'\n\s{4}\(', '\n               (', seq_string)
                seq_string = re.sub(r'\n\)', '\n           )', seq_string)
                s += f'\n    ({name}): {seq_string}'
        if len(self._layers) == 0:
            s += ')'
        else:
            s += '\n)'
        return s

    # Show the number of tensor we have in our dictionary and the number of parameters we add
    def num_params(self):
        print(f'Number of parameters: {self._num_parameters}')

    def _depth_call(self, op):
        param_list = []
        for name, value in self._parameters.items():
            if isinstance(value, Tensor):
                if op != 'get':
                    getattr(value, op)()
                else:
                    param_list.append(value)
            elif isinstance(value, dict):
                if op != 'get':
                    getattr(getattr(self, name), op)()
                else:
                    param_list.extend(getattr(self, name)._depth_call(op))
        if op == 'get':
            return param_list

    def cpu(self):
        if self.device != 'cpu':
            self.device = Device('cpu')
            self._depth_call('cpu')

    def cuda(self):
        if self.device == 'cpu':
            self.device = Device('cuda')
            self._depth_call('cuda')
