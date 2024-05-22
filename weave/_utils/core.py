import numpy as np


class HookManager:
    """
    A class to create operation hooks. These hooks serve as a point where we can revisit older values of variables that
    performed immediate operations that overwrote their own values. This way we can still traverse the operation
    graph when we do a backwards pass.

    Methods
    -----------------------------------------------
    create_hooks(hook_tensor: Tensor, op: str, alter: bool, other: Tensor = None)
        Creates a Tensor hook from the value of a 'hook_tensor'. The specific name of the hook can be controlled by the
        'op' parameter. Once the hook is created, it will be stores in the global environment of this file. 'alter'
        allows to change the attributes of the 'hook_tensor', changing its children to 'hook_tensor' or 'hook_tensor'
        and other.

    delete_hooks()
        Deletes all the operation hooks present. This is done to clear the global environment and to prevent cluttering
        the RAM with older, no more useful, variables.
    """
    this_global = globals()

    _iadd_hooks = 0
    _isub_hooks = 0
    _imul_hooks = 0
    _log_hooks = 0
    _softmax_hooks = 0
    _relu_hooks = 0

    @classmethod
    def create_hooks(cls, hook_tensor, op: str, alter: bool = False, other=None):
        match op:   # We select a base hook name based on the operation
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
        globals()[f'{name}{getattr(cls, name)}'] = hook_tensor.copy()  # We store a copy of the tenor in the hook
        if alter:
            # We change the children of the 'hook_tensor' in case we are doing an immediate operation with grad enabled
            if isinstance(other, (np.ndarray, list)) or issubclass(other, np.ndarray):
                hook_tensor._prev = {id(globals()[f'{name}{getattr(cls, name)}']), id(other)}
            else:
                hook_tensor._prev = {globals()[f'{name}{getattr(cls, name)}']}
        setattr(cls, name, getattr(cls, name) + 1)

    @classmethod
    def delete_hooks(cls):
        hooks = [arg for arg in dir(cls) if 'hooks' in arg and isinstance(getattr(cls, arg), int)]
        for name in hooks:  # We go through all hook names and all created hooks to delete them all
            for value in range(getattr(cls, name) - 1, -1, -1):
                del globals()[f'{name}{value}']
            setattr(cls, name, 0)


class MemorySet:
    """
    Represents a set that stores the ids of the elements within it as well as the elements themselves to keep references
    to them and prevent the garbage collector from deleting them.
    Implemented minimal requirements for it to be usable since only basic functionality is needed.
    """
    def __init__(self, *args):
        self.content = {}
        self._idx = 0
        for arg in args:
            if (key := id(arg)) not in self.content.keys():  # For each element stored in it, we use its id as a key
                self.content[key] = arg

    def __iter__(self):
        return self

    def __next__(self):  # Next method so we can iterate through the elements
        if self._idx < len(self.content):
            value = list(self.content.values())[self._idx]
            self._idx += 1
            return value
        raise StopIteration

    def __repr__(self):
        return f'MemorySet({self.content.__repr__()[1:-1]})'
