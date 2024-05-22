import numpy as np


class HookManager:

    this_global = globals()

    _iadd_hooks = 0
    _isub_hooks = 0
    _imul_hooks = 0
    _log_hooks = 0
    _softmax_hooks = 0
    _relu_hooks = 0

    @classmethod
    def create_hooks(cls, hook_tensor, op: str, alter: bool = False, other=None):
        match op:
            case '+':
                name = '_iadd_hooks'
            case '-':
                name = '_isub_hooks'
            case '*':
                name = '_imul_hooks'
            case 'log':
                name = '_log_hooks'
            case 'softmax':
                name = '_softmax_hooks'
            case 'relu':
                name = '_relu_hooks'
            case _:
                raise ValueError('Unsupported hook.')
        globals()[f'{name}{getattr(cls, name)}'] = hook_tensor.copy()
        if alter:
            if isinstance(other, (np.ndarray, list)) or issubclass(other, np.ndarray):
                hook_tensor._prev = {id(globals()[f'{name}{getattr(cls, name)}']), id(other)}
            else:
                hook_tensor._prev = {globals()[f'{name}{getattr(cls, name)}']}
        setattr(cls, name, getattr(cls, name) + 1)

    @classmethod
    def delete_hooks(cls):
        hooks = [arg for arg in dir(cls) if 'hooks' in arg and isinstance(getattr(cls, arg), int)]
        for name in hooks:
            for value in range(getattr(cls, name) - 1, -1, -1):
                del globals()[f'{name}{value}']
            setattr(cls, name, 0)


class MemorySet:
    def __init__(self, *args):
        self.content = {}
        self._idx = 0
        for arg in args:
            if (key := id(arg)) not in self.content.keys():
                self.content[key] = arg

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx < len(self.content):
            value = list(self.content.values())[self._idx]
            self._idx += 1
            return value
        raise StopIteration

    def __repr__(self):
        return f'MemorySet({self.content.__repr__()[1:-1]})'


if __name__ == '__main__':
    m = MemorySet([], 'a', 3242, {'sa': 4})
    print(m)
