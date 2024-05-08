from weave import Tensor
from numpy import prod


class ModelMeta(type):
    def __new__(cls, name, bases: tuple, attr: dict):
        if '__init__' in attr.keys():
            original_init = attr['__init__']

            def init_wrapper(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                self._parameter_buffer()
            attr['__init__'] = init_wrapper
        return super().__new__(cls, name, bases, attr)


class Model(metaclass=ModelMeta):
    def __init__(self):
        pass

    def _parameter_buffer(self):
        self._parameters = {}
        self._models = {}
        self._num_parameters = 0
        for attr in dir(self):
            try:
                value = getattr(self, attr)  # in order to call an attribute by using an str
            except AttributeError:
                continue  # next for iteration

            else:
                if isinstance(value, Tensor):
                    self._parameters[attr] = value
                    self._num_parameters += prod(value.shape)
                elif isinstance(value, Model):
                    self._models[attr] = value
        for layer in self._models.values():
            self._parameters.update(layer._parameters)  # we add a dictionary to the created one
            self._num_parameters += layer._num_parameters

    def params(self):
        param_dict['weights'] = list(self._parameters.values())
        param_dict['config'] = {
            'name': self.__class__.__name__,
            'num_layers': len(self._models),
            'num_parameters': len(param_dict),
        }

    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}('
        for name, value in self._models.items():
            s += f'\n    ({name}): {value}'
        if len(self._models) == 0:
            s += ')'
        else:
            s += '\n)'
        return s

    def num_params(self):
        print(f'Number of tensors: {len(self._parameters)}')
        print(f'Number of parameters: {self._num_parameters}')
