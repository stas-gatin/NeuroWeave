from typing import Any
import numpy as np
import ctypes
import re
import cupy as cp
import weave.cuda


class Tensor(np.ndarray):
    """
    Tensor class for all the arithmetic needs of the library. Inherits from numpy.ndarray to save time in recreating
    functionality.
    Adds all the necessary components for the tensor to work in the environment of a Machine Learning language.
    This class CAN be accessed by the users, but it will preferably be wrapped in other methods of this library for the
    users to have an easier understanding of how it works.
    
    [NOTE]: this class is not yet finished, but most of its features are already implemented.
    """
    def __new__(cls, shape=None, dtype=None, buffer=None, offset=0, strides=None, order=None, data=None,
                _children=(), _op=None, use_grad: bool = False, device: str | weave.cuda.Device = 'cpu'):
        # We check whether there is a shape provided or we have to infer it from the data
        if shape is None and data is not None:
            array = data.get() if isinstance(data, cp.ndarray) else np.asarray(data)
            shape = array.shape
            if dtype is None:
                dtype = array.dtype
        elif shape is None and data is None:
            raise AttributeError('Either shape or data must be given when creating the array.')
        return super().__new__(cls, shape, dtype, buffer, offset, strides, order)

    def __init__(self, shape=None, dtype=float, buffer=None, offset=0, strides=None, data=None,
                 _children=(), _op=None, use_grad: bool = False, device: str | weave.cuda.Device = 'cpu'):
        self.device = weave.cuda.Device(device) if isinstance(device, str) else device
        self._data: Any
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
        if not isinstance(data, (np.ndarray, int, float, cp.ndarray, Tensor, list)):
            raise TypeError(f'Invalid data type for Tensor: {data.__class__.__name__}')
        self[...] = data.get() if isinstance(data, cp.ndarray) else data

    def __add__(self, other: Any) -> "Tensor":
        if type(other) in [np.ndarray, list]:  # transform organized data structures into Tensors
            other = Tensor(data=other, use_grad=self._grad_enabled, device=self.device)
        elif isinstance(other, (int, float)):  # separated addition for integers and floats
            out = Tensor(data=(self.data + other), _children=(self,), _op='+', use_grad=self._grad_enabled,
                         device=self.device)

            def _backward():
                # backward pass for addition operations
                self.grad += Tensor(data=out.grad, device=self.device)

            if self._grad_enabled:  # only include backward function if the grad is enabled
                out._backward = _backward
            return out
        elif not isinstance(other, Tensor):
            raise TypeError(f"Cannot add 'Tensor' and {type(other)} objects.")

        assert self.device == other.device, 'Expected both tensors to be on the same device, but found two: '\
                                            f'{self.device} and {other.device}.'
        # We relay on NumPy to handle the actual arithmetic operations
        out = Tensor(data=(self.data + other.data), _children=(self, other), _op='+', use_grad=self._grad_enabled,
                     device=self.device)

        def _backward():
            # backward pass for addition operations (gradient flows toward the children)
            self.grad += Tensor(data=out.grad, device=self.device)
            other.grad += Tensor(data=out.grad, device=other.device)

        if self._grad_enabled:
            out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __radd__(self, other: Any) -> "Tensor":
        return self + other

    def __iadd__(self, other: Any) -> "Tensor":
        out = self.__add__(other)
        self.grad = out.grad
        self.data = out.data
        self._populate(out.data)
        self._prev = out._prev
        self._op = out._op
        return self

    def __sub__(self, other: Any) -> "Tensor":
        return self + (-other)

    def __rsub__(self, other: Any) -> "Tensor":
        return (-self) + other

    def __isub__(self, other: Any) -> "Tensor":
        out = self + (-other)
        self.grad = out.grad
        self.data = out.data
        self._populate(out.data)
        self._prev = out._prev
        self._op = out._op
        return self

    def __mul__(self, other: Any) -> "Tensor":
        if type(other) in [np.ndarray, list]:
            other = Tensor(data=other, use_grad=self._grad_enabled ,device=self.device)
        elif isinstance(other, (int, float)):  # specialized output for mul of Tensor and a number
            out = Tensor(data=(self.data * other), _children=(self,), _op='*', use_grad=self._grad_enabled,
                         device=self.device)

            def _backward():
                # backward pass using the chain rule for multiplication
                self.grad += other * out.grad

            if self._grad_enabled:
                out._backward = _backward
            return out
        elif not isinstance(other, Tensor):
            raise TypeError(f"Cannot multiply 'Tensor' and {type(other)} objects.")

        assert self.device == other.device, 'Expected both tensors to be on the same device, but found two: ' \
                                            f'{self.device} and {other.device}.'
        out = Tensor(data=(self.data * other.data), _children=(self, other), _op='*', use_grad=self._grad_enabled,
                     device=self.device)

        def _backward():
            # backward pass using the chain rule for multiplication
            self._grad_enabled = False
            other._grad_enabled = False
            self.grad += other * out.grad
            other.grad += self * out.grad
            self._grad_enabled = True
            other._grad_enabled = True

        if self._grad_enabled:
            out._backward = _backward
        return out

    def __rmul__(self, other: Any) -> "Tensor":
        return self * other

    def __imul__(self, other: Any) -> "Tensor":
        out = self * other
        self.grad = out.grad
        self.data = out.data
        self._populate(out.data)
        self._prev = out._prev
        self._op = out._op
        return self

    def __pow__(self, other: Any) -> "Tensor":
        if type(other) in [np.ndarray, list]:
            other = Tensor(data=other, use_grad=self._grad_enabled, device=self.device)
        elif isinstance(other, (int, float)):
            this = self.data
            this.dtype = float
            out = Tensor(data=(this ** other), _children=(self,), _op='**', use_grad=self._grad_enabled,
                         device=self.device)

            def _backward():
                # backward pass making use of the derivative of a power (d/dx x**2 = 2*x)
                self._grad_enabled = False
                self.grad += other * self ** (other - 1) * out.grad
                self._grad_enabled = True

            if self._grad_enabled:
                out._backward = _backward
            return out
        elif not isinstance(other, Tensor):
            raise TypeError(f"Cannot raise Tensor to the power of a {type(other)}")

        assert self.device == other.device, 'Expected both tensors to be on the same device, but found two: ' \
                                            f'{self.device} and {other.device}.'
        out = Tensor(data=(self.data ** other.data), _children=(self, other), _op='**', use_grad=self._grad_enabled,
                     device=self.device)

        def _backward():
            self._grad_enabled = False
            other._grad_enabled = False
            # backward pass making use of the partial derivative of powers
            self.grad += other * self ** (other - 1) * out.grad  # d/dx x**y = y*x**(y - 1)
            other.grad += (self ** other) * self.log() * out.grad  # d/dy x**y = (x**y) * ln(x)
            self._grad_enabled = True
            other._grad_enabled = True

        if self._grad_enabled:
            out._backward = _backward
        return out

    def __rpow__(self, other: Any) -> "Tensor":
        # We have to use different calculations since 2**x != x**2
        if type(other) in [np.ndarray, list]:
            return Tensor(data=other, use_grad=self._grad_enabled, device=self.device) ** self
        elif isinstance(other, (int, float)):
            out = Tensor(data=(other ** self.data), _children=(self,), _op='**', use_grad=self._grad_enabled,
                         device=self.device)

            def _backward():
                # since this is n**x (n a number), we use d/dx n**x = (n**x) * ln(n)
                self._grad_enabled = False
                self.grad += (other ** self) * np.log(other) * out.grad
                self._grad_enabled = True

            if self._grad_enabled:
                out._backward = _backward
            return out
        elif not isinstance(other, Tensor):
            raise TypeError(f"Cannot raise {type(other)} to the power of a Tensor.")
        return other ** self

    def __ipow__(self, other: Any) -> "Tensor":
        out = self ** other
        self.grad = out.grad
        self.data = out.data
        self._populate(out.data)
        self._prev = out._prev
        self._op = out._op
        return self

    def __truediv__(self, other: Any) -> "Tensor":
        if type(other) in [np.ndarray, list]:
            other = Tensor(data=other, use_grad=self._grad_enabled, device=self.device)
        elif isinstance(other, (int, float)):  # for numbers, we invert them and multiply instead of dividing
            return self * (other ** -1)
        elif not isinstance(other, Tensor):
            raise TypeError(f"Cannot divide Tensor by object of type {type(other)}.")

        assert self.device == other.device, 'Expected both tensors to be on the same device, but found two: ' \
                                            f'{self.device} and {other.device}.'
        out = Tensor(data=(self.data / other.data), _children=(self, other), _op='/', use_grad=self._grad_enabled,
                     device=self.device)

        def _backward():
            self._grad_enabled = False
            other._grad_enabled = False
            # We use the partial derivatives for the division of variables
            self.grad += out.grad / other  # d/dx x/y = 1/y
            other.grad += -((self / other ** 2) * out.grad)  # d/dy x/y = x/(y**2)
            self._grad_enabled = True
            other._grad_enabled = True

        if self._grad_enabled:
            out._backward = _backward
        return out

    def __rtruediv__(self, other: Any) -> "Tensor":
        if type(other) in [np.ndarray, list]:
            other = Tensor(data=other, use_grad=self._grad_enabled, device=self.device)
            return other / self
        elif isinstance(other, (int, float)):
            return (self ** -1) * other
        else:
            raise TypeError(f"Cannot divide Tensor by object of type {type(other)}.")

    def __itruediv__(self, other: Any) -> "Tensor":
        out = self / other
        self.grad = out.grad
        self.data = out.data
        self._populate(out.data)
        self._prev = out._prev
        self._op = out._op
        return self

    def __floordiv__(self, other: Any) -> "Tensor":
        raise NotImplementedError("Floor division not implemented for tensors.")

    def __rfloordiv__(self, other: Any) -> "Tensor":
        raise NotImplementedError("Floor division not implemented for tensors.")

    def __ifloordiv__(self, other: Any) -> "Tensor":
        raise NotImplementedError("Floor division not implemented for tensors.")

    def __matmul__(self, other: Any) -> "Tensor":
        # Usual matrix multiplication for tensors
        if type(other) in [np.ndarray, list]:
            other = Tensor(data=other, use_grad=self._grad_enabled, device=self.device)
        elif not isinstance(other, Tensor):  # don't allow multiplication with scalars
            raise TypeError(f"Cannot multiply 'Tensor' and {type(other)}")

        assert self.device == other.device, 'Expected both tensors to be on the same device, but found two: ' \
                                            f'{self.device} and {other.device}.'
        out = Tensor(data=(self.data @ other.data), _children=(self, other), _op='@', use_grad=self._grad_enabled,
                     device=self.device)

        def _backward():
            # According to calculations done in PyTorch, there are two formulas to calculate the grad of a matmul
            # 1. out.grad @ (d parent/d child).T
            # 2. (d parent/d child).T @ out.grad
            # We have no way of knowing which one is the correct one, so we have to try both and correct if the
            # operation cannot be performed or if the resulting shape is incorrect. What we do know is that we have to
            # do the other operation after we performed one of them, so if one child did 1. we have to do 2. next and
            # vice versa.
            obj_list = [self, other]
            jump = False
            for i, current in enumerate(obj_list):
                following = obj_list[((i + 1) % 2)]
                try:
                    # 1.
                    if jump: raise ValueError
                    res = out.grad.data @ following.data.T
                    assert res.shape == current.shape, 'Error1'
                    current.grad += Tensor(data=res, device=out.device)
                    jump = True
                except (ValueError, AssertionError):
                    # 2.
                    res = following.data.T @ out.grad.data
                    current.grad += Tensor(data=res, device=out.device)

        if self._grad_enabled:
            out._backward = _backward
        return out

    def __rmatmul__(self, other: Any) -> "Tensor":
        if type(other) in [np.ndarray, list]:
            other = Tensor(data=other, use_grad=self._grad_enabled, device=self.device)
            return other @ self
        else:
            raise TypeError(f"Cannot multiply 'Tensor' and {type(other)}.")

    def __imatmul__(self, other: Any) -> "Tensor":
        out = self @ other
        self.grad = out.grad
        self.data = out.data
        self._populate(out.data)
        self._prev = out._prev
        self._op = out._op
        return self

    def log(self):
        out = Tensor(data=np.log(self.data), _children=(self,), _op='log', use_grad=self._grad_enabled,
                     device=self.device)

        def _backward():
            self._grad_enabled = False
            self.grad += (1 / Tensor(data=self.data, device=self.device)) * out.grad  # d/dx ln(x) = 1/x
            self._grad_enabled = True

        if self._grad_enabled:
            out._backward = _backward
        return out

    @property
    def data(self) -> np.ndarray | cp.ndarray:
        # We have to reset the 'data' property since NumPy already makes use of it
        return self._data

    @data.setter
    def data(self, value):
        if self.device == 'cpu':
            if value is None:
                self._data = np.asarray(self)
            else:
                self._data = value.get() if isinstance(value, cp.ndarray) else np.asarray(value)
        else:
            if value is None:
                self._data = cp.asarray(self)
            else:
                self._data = value if isinstance(value, cp.ndarray) else cp.asarray(value)

    def backward(self):
        # Backpropagation algorithm to traverse the operation graph that we have built with each mathematical operation
        topo = []
        visited = set()

        def build_topo(t: "Tensor"):
            # We build a function to perform a topological sort on all the tensors involved in the creation of the
            # tensor that the backward was called on.
            if id(t) not in visited:
                visited.add(id(t))
                for child in t._prev:
                    build_topo(ctypes.cast(child, ctypes.py_object).value)
                topo.append(t)

        build_topo(self)
        [print(t) for t in reversed(topo)]
        self.grad = (Tensor(self.shape, device=self.device) * 0) + 1  # We set the first tensor's gradient to 1
        for node in reversed(topo):
            node._backward()  # Run the backwards function of all tensors in reverse

    def cpu(self):
        if self.device != 'cpu':
            self.data = self.data.get()

    def cuda(self):
        if self.device == 'cpu':
            self.data = cp.asarray(self.data)

    def __getitem__(self, idx):
        s = np.asarray(self.data)
        if isinstance(idx, (int, slice)):
            return Tensor(data=s[idx])
        elif isinstance(idx, tuple):
            try:  # manage in case we are indexing a number
                r = float(s[idx])
            except TypeError:
                r = s[idx]
            return Tensor(data=r)
        elif isinstance(idx, (list, np.ndarray)):
            return Tensor(data=s[tuple(idx)])
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def __str__(self):
        info = self.data.get() if isinstance(self.data, cp.ndarray) else self.data
        data_string = np.asarray(info, dtype=self.dtype).__repr__()
        data_string = data_string[6:-1].rsplit('\n') if 'array' in data_string else data_string.rsplit('\n')
        data_string = [data_string[0]] + [' ' + line.strip() for line in data_string[1:]]
        data_string = '\n'.join(data_string)
        return data_string

    def __repr__(self):
        # Display the Tensors in a way that is consistent to how NumPy, PyTorch, Tensorflow and the such do.
        info = self.data.get() if isinstance(self.data, cp.ndarray) else self.data
        data_string = np.asarray(info, dtype=self.dtype).__repr__()
        data_string = re.sub(r',\s\[', ',\n       [', data_string)
        data_string = data_string[6:-1].rsplit('\n') if 'array' in data_string else data_string.rsplit('\n')
        data_string = [data_string[0]] + [' ' + line for line in data_string[1:]]
        data_string = '\n'.join(data_string)
        s = f'Tensor({data_string}, dtype={self.dtype}' if not self._grad_enabled else f'Tensor({data_string}, ' \
                                                                                        f'dtype={self.dtype}, ' \
                                                                                        f'uses_grad={self._grad_enabled}'
        if self.device != 'cpu':
            s += f', device={self.device}'
        s += ')'
        return s


if __name__ == '__main__':
    a = Tensor(data=[[1, 2], [3, 4]], dtype=float, use_grad=True, device='cuda')
    b = Tensor(data=[[5, 6], [7, 8]], dtype=float, use_grad=True, device='cuda')
    c = Tensor(data=[[9, 8]], dtype=float, use_grad=True, device='cuda')
    d = Tensor(data=[[3], [8]], dtype=float, use_grad=True, device='cuda')
    e = a - b
    f = c @ e
    g = f @ d
    print(g)
    g.backward()
    print(g.grad.__repr__())
    print(d.grad.__repr__())
    print(f.grad.__repr__())
    print(c.grad.__repr__())
    print(e.grad.__repr__())
    print(a.grad.__repr__())
    print(b.grad.__repr__())
