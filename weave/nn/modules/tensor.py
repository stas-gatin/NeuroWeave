import random
from typing import Any
import numpy as np
import ctypes  # Mirar la posición de memoria donde están guardados los objetos a través de la posición
import time


# [NOTE]: Cuando __init__ y __new__ están juntos, __new__ se ejecuta primero
class Tensor(np.ndarray):
    # Heredamos de la clase array de numpy(father)
    """
    Tensor class for all the arithmetic needs of the library. Inherits from numpy.ndarray to save time in recreating functionality.
    Adds all the necesary components for the tensor to work in the environment of a Machine Learning language.
    This class CAN be accessed by the users but it will preferably be wraped in other methods of this library for the users to have
    an easier understanding of how it works.

    [NOTE]: this class is not yet finished, all current capabilities are susceptible to change, same goes for this docstring.
    """

    # data: lista de listas (array)
    # _children: relación contraria padre-hijo('s), elemento necesario en la suma para poder crear e
    # _op: muestra las operaciones que hemos realizado para hacer visualizaciones
    def __new__(cls, shape=None, dtype=float, buffer=None, offset=0, strides=None, order=None, data=None,
                _children=(), _op=None):
        # Si no tenemos la forma de la matriz, pero si sus datos, la creamos y verificamos también que la información
        # que nos han proporcionado sea la adecuada
        if shape is None and data is not None:
            array = np.asarray(data)
            shape = array.shape
            dtype = array.dtype
        elif shape is None and data is None:
            raise AttributeError('Either shape or data must be given when creating the array.')

        # Devuelve un tensor
        return super().__new__(cls, shape, dtype, buffer, offset, strides, order)

    # Aunque no utilicemos los argumentos, tenemos que pasarlos porque si no, el __new__ no los reconoce
    def __init__(self, shape=None, dtype=float, buffer=None, offset=0, strides=None, data=None,
                 _children=(), _op=None):
        self.data = data
        if data is not None:
            self._populate(data)
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(id(child) for child in _children)
        self._op = _op

    def __array_finalize__(self, obj):
        pass

    def _populate(self, data):
        # Para cada dimensión, a partir de la primera posición
        slicing = [slice(None, None, None) for _ in enumerate(self.shape[1:])]
        for i in range(self.shape[-1]):
            start = [slice(i, None, None)] + slicing
            self[*start] = data[i]

    def __add__(self, other: Any) -> "Tensor":
        if type(other) in [np.ndarray, list]:
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

    def __rmul__(self, other: Any) -> "Tensor":
        return self * other

    def __imul__(self, other: Any) -> "Tensor":
        return self * other

    def __pow__(self, other: Any) -> "Tensor":
        assert isinstance(other, (int, float)), 'Powers of types other than int and float are not supported.'
        out = Tensor(data=(np.asarray(self) ** other), _children=(self,), _op=f'**{other}')

        def _backward():
            self.grad += other * Tensor(data=self.data) ** (other - 1) * out.grad

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
        out = Tensor(data=(np.asarray(self) @ np.asarray(other)), _children=(self, other), _op='@')

        # TODO: Find how to calculate the backwards for usual matrix multiplication
        def _backward():
            pass

        out.backward = _backward
        return out

    # Decoradores: hacer que una función se comporte como un atributo
    # Queremos los datos y por eso hemos creado una propiedad
    # Si tenemos una propiedad y queremos darle valores, tenemos que poner 'X(propiedad).setter'
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
            if id(t) not in visited:
                visited.add(id(t))
                for child in t._prev:
                    build_topo(ctypes.cast(child, ctypes.py_object).value)
                topo.append(t)

        build_topo(self)
        self.grad = (Tensor(self.shape) * 0) + 1
        for node in reversed(topo):
            node._backward()
