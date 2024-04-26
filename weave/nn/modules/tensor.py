from typing import Any
import numpy as np
import ctypes
import re


# [NOTE]: Cuando __init__ y __new__ están juntos, __new__ se ejecuta primero
class Tensor(np.ndarray):
    """
    Tensor class for all the arithmetic needs of the library. Inherits from numpy.ndarray to save time in recreating
    functionality.
    Adds all the necessary components for the tensor to work in the environment of a Machine Learning language.
    This class CAN be accessed by the users, but it will preferably be wrapped in other methods of this library for the
    users to have an easier understanding of how it works.
    
    [NOTE]: this class is not yet finished, all current capabilities are susceptible to change,
    same goes for this docstring.
    """
    def __new__(cls, shape=None, dtype=None, buffer=None, offset=0, strides=None, order=None, data=None,
                _children=(), _op=None, use_grad: bool = False):
        if shape is None and data is not None:
            array = np.asarray(data)
            shape = array.shape
            if dtype is None:
                dtype = array.dtype
        elif shape is None and data is None:
            raise AttributeError('Either shape or data must be given when creating the array.')
        return super().__new__(cls, shape, dtype, buffer, offset, strides, order)

    def __init__(self, shape=None, dtype=float, buffer=None, offset=0, strides=None, data=None,
                 _children=(), _op=None, use_grad: bool = False):
        self.data = data
        if data is not None:
            self._populate(data)
        self.grad = 0
        if use_grad:
            self._backward = lambda: None
        self._prev = set(id(child) for child in _children)
        self._op = _op
        self._grad_enabled = use_grad

    def __array_finalize__(self, obj):
        pass

    def _populate(self, data):
        slicing = [slice(None, None, None) for _ in enumerate(self.shape[1:])]
        for i in range(self.shape[0]):
            self[i, *slicing] = data[i]

    def __add__(self, other: Any) -> "Tensor":
        if type(other) in [np.ndarray, list]:
            other = Tensor(data=other)
        elif isinstance(other, int):
            return Tensor(data=(np.asarray(self) + other))
        elif not isinstance(other, Tensor):
            raise TypeError(f"Cannot add 'Tensor' and {type(other)} objects.")

        out = Tensor(data=(np.asarray(self) + np.asarray(other)), _children=(self, other), _op='+',
                     use_grad=self._grad_enabled)

        def _backward():
            self.grad += Tensor(data=out.grad)
            other.grad += Tensor(data=out.grad)

        if self._grad_enabled:
            out._backward = _backward
        return out

    def __radd__(self, other: "Tensor") -> "Tensor":
        return self + other

    def __iadd__(self, other: "Tensor") -> "Tensor":
        return self + other

    def __sub__(self, other: "Tensor") -> "Tensor":
        return self + (-other)

    def __rsub__(self, other: "Tensor") -> "Tensor":
        return self + (-other)

    def __isub__(self, other: "Tensor") -> "Tensor":
        return self + (-other)

    def __mul__(self, other: Any) -> "Tensor":
        if type(other) in [np.ndarray, list]:
            other = Tensor(data=other)
        elif isinstance(other, int):
            return Tensor(data=(np.asarray(self) * other))
        elif not isinstance(other, Tensor):
            raise TypeError(f"Cannot multiply 'Tensor' and {type(other)} objects.")
        out = Tensor(data=(np.asarray(self) * np.asarray(other)), _children=(self, other), _op='*',
                     use_grad=self._grad_enabled)

        def _backward():
            self.grad += Tensor(data=other._data) * Tensor(data=out.grad)
            other.grad += Tensor(data=self._data) * Tensor(data=out.grad)

        if self._grad_enabled:
            out._backward = _backward
        return out

    def __rmul__(self, other: Any) -> "Tensor":
        return self * other

    def __imul__(self, other: Any) -> "Tensor":
        return self * other

    def __pow__(self, other: Any) -> "Tensor":
        assert isinstance(other, (int, float)), 'Powers of types other than int and float are not supported.'
        out = Tensor(data=(np.asarray(self) ** other), _children=(self,), _op=f'**{other}',
                     use_grad=self._grad_enabled)

        def _backward():
            self.grad += other * Tensor(data=self.data) ** (other - 1) * out.grad

        if self._grad_enabled:
            out._backward = _backward
        return out

    def __truediv__(self, other: Any) -> "Tensor":
        raise NotImplemented("__truediv__ is not yet implemented.")
        # if type(other) in [np.ndarray, list]:
        #     other = Tensor(data=other)
        # elif isinstance(other, int):
        #     return Tensor(data=(np.asarray(self) / other))
        # elif not isinstance(other, Tensor):
        #     raise TypeError(f"Cannot divide 'Tensor' and {type(other)} objects.")
        # out = Tensor(data=(np.asarray(self) / np.asarray(other)), _children=(self, other), _op='/')

    def __matmul__(self, other: Any) -> "Tensor":
        if type(other) in [np.ndarray, list]:
            other = Tensor(data=other)
        elif not isinstance(other, Tensor):
            raise TypeError(f"Cannot multiply 'Tensor' and {type(other)}")
        out = Tensor(data=(np.asarray(self) @ np.asarray(other)), _children=(self, other), _op='@',
                     use_grad=self._grad_enabled)

        def _backward():
            obj_list = [self, other]
            for i, current in enumerate(obj_list):
                following = obj_list[((i + 1) % 2)]
                # print('Inside backward:')
                # print('\t', np.asarray(out.grad))
                # print('\t', np.asarray(following.data).T)
                try:
                    res = np.asarray(out.grad) @ np.asarray(following.data).T
                    assert res.shape == current.shape, 'Error'
                    current.grad += Tensor(data=res)
                except ValueError or AssertionError:
                    res = np.asarray(following.data).T @ np.asarray(out.grad)
                    assert res.shape == current.shape, 'Error'
                    current.grad += Tensor(data=res)

        if self._grad_enabled:
            out._backward = _backward
        return out

    def __rmatmul__(self, other: "Tensor") -> "Tensor":
        return self.__matmul__(other)

    def __imatmul__(self, other: "Tensor") -> "Tensor":
        return self.__matmul__(other)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @data.getter
    def data(self):
        return self._data

    def backward(self):
        topo = []
        visited = set()

        def build_topo(t: "Tensor"):
            if id(t) not in visited:
                visited.add(id(t))
                for child in t._prev:
                    build_topo(ctypes.cast(child, ctypes.py_object).value)
                topo.append(t)

        build_topo(self)

        self.grad = (Tensor(self.shape) * 0) + 1
        for node in reversed(topo):
            node._backward()

    def __str__(self):
        data_string = self.data.__repr__()
        data_string = re.sub(r',\s\[', ',\n       [', data_string)
        data_string = data_string[6:-1].rsplit('\n') if 'array' in data_string else data_string.rsplit('\n')
        data_string = [data_string[0]] + [' ' + line for line in data_string[1:]]
        data_string = '\n'.join(data_string)
        s = f'Tensor({data_string}, dtype={self.dtype})' if not self._grad_enabled else f'Tensor({data_string}, '\
                                                                                      f'dtype={self.dtype}, '\
                                                                                    f'uses_grad={self._grad_enabled})'
        return s

    def __repr__(self):
        return f"Tensor({self.data})"
