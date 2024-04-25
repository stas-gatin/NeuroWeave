import random
from typing import Any
import numpy as np
import ctypes


# [NOTE]: Cuando __init__ y __new__ están juntos, __new__ se ejecuta primero
class Tensor(np.ndarray):
    """
    Tensor class for all the arithmetic needs of the library. Inherits from numpy.ndarray to save time in recreating functionality.
    Adds all the necesary components for the tensor to work in the environment of a Machine Learning language.
    This class CAN be accessed by the users but it will preferably be wraped in other methods of this library for the users to have
    an easier understanding of how it works.
    
    [NOTE]: this class is not yet finished, all current capabilities are susceptible to change, same goes for this docstring.
    """
    def __new__(cls, shape=None, dtype=float, buffer=None, offset=0, strides=None, order=None, data=None,
                _children=(), _op=None):
        if shape is None and data is not None:
            shape = np.asarray(data).shape
        elif shape is None and data is None:
            raise AttributeError('Either shape or data must be given when creating the array.')
        return super().__new__(cls, shape, dtype, buffer, offset, strides, order)

    def __init__(self, shape=None, dtype=float, buffer=None, offset=0, strides=None, data=None,
                 _children=(), _op=None):
        if data is not None:
            self._populate(data)
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(id(child) for child in _children)
        self._op = _op

    def __array_finalize__(self, obj):
        pass

    def _populate(self, data):
        for i in range(self.shape[0]):
            self[i, :] = data[i]

    def __add__(self, other: Any) -> "Tensor":
        if isinstance(other, np.ndarray | list):
            other = Tensor(data=other)
        elif isinstance(other, int):
            return Tensor(data=(np.asarray(self) + other))
        elif not isinstance(other, Tensor):
            raise TypeError(f"Cannot add 'Tensor' and {type(other)} objects.")

        out = Tensor(data=(np.asarray(self) + np.asarray(other)), _children=(self, other), _op='+')

        def _backward():
            self.grad += Tensor(data=out.grad)
            other.grad += Tensor(data=out.grad)

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
        out = Tensor(data=(np.asarray(self) * np.asarray(other)), _children=(self, other), _op='*')

        def _backward():
            self.grad += Tensor(data=other._data) * Tensor(data=out.grad)
            other.grad += Tensor(data=self._data) * Tensor(data=out.grad)

        out._backward = _backward
        return out

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def backward(self):
        topo = []
        visited = set()

        def build_topo(t: "Tensor"):
            print(t)
            if id(t) not in visited:
                visited.add(id(t))
                for child in t._prev:
                    build_topo(ctypes.cast(child, ctypes.py_object).value)
                topo.append(t)

        build_topo(self)
        self.grad = (Tensor(self.shape) * 0) + 1
        for node in reversed(topo):
            node._backward()
