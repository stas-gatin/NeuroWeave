from weave import Tensor
from weave.cuda import Device


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

    def forward(self, x: Tensor) -> Tensor:
        pass

    def __call__(self, x: Tensor) -> Tensor:
        pass

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

    def params(self):
        param_dict = {'weights': [self._parameters]}
        param_dict['config'] = {
            'name': self.__class__.__name__,
            'num_layers': len(self._layers),
            'num_parameters': len(param_dict),
        }
        return param_dict

    # Load values from a trained model
    @classmethod
    def load(cls, values: dict, *args):
        m = cls(*args)  # classmethod is empty

        def rec_fill(obj, weights):
            obj._parameters = weights
            obj._layers = {}
            for name, value in weights.items():
                if isinstance(value, Tensor):
                    setattr(obj, name, value)
                elif isinstance(value, dict):
                    obj._layers[name] = getattr(obj, name)
                    rec_fill(getattr(obj, name), value)
        rec_fill(m, values)
        return m

    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}('
        for name, value in self._layers.items():
            s += f'\n    ({name}): {value}'
        if len(self._layers) == 0:
            s += ')'
        else:
            s += '\n)'
        return s

    # Show the number of tensor we have in our dictionary and the number of parameters we add
    def num_params(self):
        print(f'Number of tensors: {len(self._parameters)}')
        print(f'Number of parameters: {self._num_parameters}')

    def _depth_call(self, op):
        for name, value in self._parameters.items():
            if isinstance(value, Tensor):
                getattr(value, op)()
            elif isinstance(value, dict):
                getattr(self, name)._depth_call(op)

    def cpu(self):
        if self.device != 'cpu':
            self.device = Device('cpu')
            self._depth_call('cpu')

    def cuda(self):
        if self.device == 'cpu':
            self.device = Device('cuda')
            self._depth_call('cuda')
