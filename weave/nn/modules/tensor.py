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
        # We check whether there is a shape provided or we have to infer it from the data
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
            self._populate(data)  # If data was provided, we populate the tensor making use of NumPy's API
        self.grad = 0
        # We only store the data for the backward functions if use_grad is enabled
        if use_grad:
            self._backward = lambda: None
        # We transform the Tensor objects to their ids to make them hashable and store them into a set
        self._prev = set(id(child) for child in _children)
        self._op = _op
        self._grad_enabled = use_grad

    def __array_finalize__(self, obj):
        # We just make use of NumPy's API, but we have to adapt it in order to get a Tensor object
        pass

    def _populate(self, data):
        slicing = [slice(None, None, None) for _ in enumerate(self.shape[1:])]  # Index all dimensions after the first
        for i in range(self.shape[0]):
            # Requires Python 3.10+
            self[i, *slicing] = data[i]  # Fill the Tensor indexing the first dimension and filling all others

    def __add__(self, other: Any) -> "Tensor":
        if type(other) in [np.ndarray, list]:
            other = Tensor(data=other, dtype=self.dtype, use_grad=self._grad_enabled)
        elif isinstance(other, (int, float)):
            out = Tensor(data=(np.asarray(self) + other), _children=(self,), _op='+', use_grad=self._grad_enabled)

            def _backward():
                self.grad += Tensor(data=out.grad)

            if self._grad_enabled:
                out._backward = _backward
            return out
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

    def __radd__(self, other: Any) -> "Tensor":
        return self + other

    def __iadd__(self, other: Any) -> "Tensor":
        return self + other

    def __sub__(self, other: Any) -> "Tensor":
        return self + (-other)

    def __rsub__(self, other: Any) -> "Tensor":
        return (-self) + other

    def __isub__(self, other: Any) -> "Tensor":
        return self + (-other)

    def __mul__(self, other: Any) -> "Tensor":
        if type(other) in [np.ndarray, list]:
            other = Tensor(data=other)
        elif isinstance(other, (int, float)):
            out = Tensor(data=(np.asarray(self) * other), _children=(self,), _op='*', use_grad=self._grad_enabled)

            def _backward():
                self.grad += other * Tensor(data=out.grad)

            if self._grad_enabled:
                out._backward = _backward
            return out
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
        if type(other) in [np.ndarray, list]:
            other = Tensor(data=other)
        elif isinstance(other, (int, float)):
            out = Tensor(data=(np.asarray(self) ** other), _children=(self,), _op='**', use_grad=self._grad_enabled)

            def _backward():
                self.grad += other * Tensor(data=self.data) ** (other - 1) * out.grad

            if self._grad_enabled:
                out._backward = _backward
            return out
        elif not isinstance(other, Tensor):
            raise TypeError(f"Cannot raise Tensor to the power of a {type(other)}")
        out = Tensor(data=(np.asarray(self) ** np.asarray(other)), _children=(self, other), _op='**',
                     use_grad=self._grad_enabled)

        def _backward():
            self.grad += other * Tensor(data=self.data) ** (other - 1) * out.grad
            other.grad += self * Tensor(data=other.data) ** (self - 1) * out.grad

        if self._grad_enabled:
            out._backward = _backward
        return out

    def __rpow__(self, other: Any) -> "Tensor":
        if type(other) in [np.ndarray, list]:
            return Tensor(data=other, use_grad=self._grad_enabled) ** self
        elif isinstance(other, (int, float)):
            out = Tensor(data=(other ** np.asarray(self)), _children=(self,), _op='**', use_grad=self._grad_enabled)

            def _backward():
                self.grad = other * Tensor(data=self.data) ** (other - 1) * out.grad

            if self._grad_enabled:
                out.backward = _backward
            return out
        elif not isinstance(other, Tensor):
            raise TypeError(f"Cannot raise {type(other)} to the power of a Tensor.")
        return other ** self

    def __ipow__(self, other: Any) -> "Tensor":
        return self ** other

    def __truediv__(self, other: Any) -> "Tensor":
        if type(other) in [np.ndarray, list]:
            other = Tensor(data=other, use_grad=self._grad_enabled)
        elif isinstance(other, (int, float)):
            return self * (other ** -1)
        elif not isinstance(other, Tensor):
            raise TypeError(f"Cannot divide Tensor by object of type {type(other)}.")
        out = Tensor(data=(np.asarray(self) / np.asarray(other)), _children=(self, other), _op='/',
                     use_grad=self._grad_enabled)

    def __rtruediv__(self, other: Any) -> "Tensor":
        return self.__mul__(other ** -1)

    def __itruediv__(self, other: Any) -> "Tensor":
        return self.__mul__(other ** -1)

    def __floordiv__(self, other: Any) -> "Tensor":
        raise NotImplementedError("Floor division not implemented yet.")

    def __rfloordiv__(self, other: Any) -> "Tensor":
        raise NotImplementedError("Floor division not implemented yet.")

    def __ifloordiv__(self, other: Any) -> "Tensor":
        raise NotImplementedError("Floor division not implemented yet.")

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
                # print(i, 'Inside backward:')
                # print('\t', np.asarray(out.grad))
                # print('\t', np.asarray(following.data).T)
                try:
                    res = np.asarray(out.grad) @ np.asarray(following.data).T
                    assert res.shape == current.shape, 'Error1'
                    current.grad += Tensor(data=res)
                except (ValueError, AssertionError):
                    res = np.asarray(following.data).T @ np.asarray(out.grad)
                    assert res.shape == current.shape, 'Error2'
                    current.grad += Tensor(data=res)

        if self._grad_enabled:
            out._backward = _backward
        return out

    def __rmatmul__(self, other: Any) -> "Tensor":
        return self.__matmul__(other)

    def __imatmul__(self, other: Any) -> "Tensor":
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


if __name__ == '__main__':
    a = Tensor(data=[[1, 2], [3, 4]], dtype=float, use_grad=True)
    b = Tensor(data=[[5, 6], [7, 8]], dtype=float, use_grad=True)
    print(a / [[5, 6], [7, 8]])
