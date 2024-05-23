"""
MIT License

Copyright (c) 2024 NeuroWeave

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Any, Union
import numpy as np
import re
from weave._utils import HookManager, MemorySet  # import necessary classes to manage the garbage collector.
import weave.cuda  # import cuda to allow Tensors to live on the GPU
if weave.cuda.is_available():
    import cupy as cp  # we still require CuPy for transformations


class Tensor(np.ndarray):
    """
    Tensor class for all the arithmetic needs of the library. Inherits from numpy.ndarray to save time in recreating
    functionality.
    Allows multidimensional arrays to be used for a wide range of mathematical operations, as well as allowing said
    arrays to be stored in either the CPU or a CUDA compatible GPU.

    Attributes
    -----------------------------------------------
    data : numpy.ndarray | cupy.ndarray
        The data contained in the tensor. It can be set put is preferable not to do so.
    shape: tuple
        The size of the tensor. Inherited from numpy.ndarray
    grad : Tensor
        The gradient of the tensor after applying the backpropagation algorithm. Initialized at 0.
    device: weave.cuda.Device
        An object that represents in which device the tensor is allocated. Can be changed with Tensor.cpu() or
        Tensor.cuda().
    """

    def __new__(cls, shape=None, dtype=None, buffer=None, offset=0, strides=None, order=None, data=None,
                _children=(), _op=None, use_grad: bool = False, device: Union['weave.cuda.Device', str] = 'cpu'):
        # We check whether there is a shape provided or we have to infer it from the data
        if shape is None and data is not None:
            # Handle differences between data on the CPU and the GPU
            if device == 'cpu' or (isinstance(device, str) and 'cpu' in device):
                array = np.asarray(data)
            else:
                array = data.get() if isinstance(data, cp.ndarray) else np.asarray(data)
            shape = array.shape
            if dtype is None:
                dtype = array.dtype
        elif shape is None and data is None:
            raise AttributeError('Either shape or data must be given when creating the array.')
        return super().__new__(cls, shape, dtype, buffer, offset, strides, order)

    def __init__(self, shape=None, dtype=None, *, data=None, _children=(), _op=None, use_grad: bool = False,
                 device: Union['weave.cuda.Device', str] = 'cpu'):
        if use_grad is True and self.dtype not in [*weave._float_types, float]:
            raise ValueError('Only Tensors with floating types support gradients.')
        self.device = weave.cuda.Device(device) if isinstance(device, str) else device  # assign the correct device
        self._data: Any
        self.data = data
        if data is not None:
            self._populate(data)  # If data was provided, we populate the tensor making use of NumPy's API
        self.grad = 0
        # We only store the data for the backward functions if use_grad is enabled
        if use_grad:
            self._backward = lambda: None
            self._prev = MemorySet(*[child for child in _children])
        self._op = _op
        self._grad_enabled = use_grad

    def __array_finalize__(self, obj):
        # We just make use of NumPy's API, but we have to adapt it in order to get a Tensor object
        pass

    def _populate(self, data):
        if self.device == 'cpu':
            cpu_accepted_types = Union[np.ndarray, int, float, Tensor, list, *weave._types]
            if not isinstance(data, cpu_accepted_types):
                raise TypeError(f'Invalid data type for Tensor: {data.__class__.__name__}')
            # By using ellipsis indexing, we can refer to all the dimensions of the Tensor
            self[...] = data
        else:
            cuda_accepted_types = Union[np.ndarray, int, float, cp.ndarray, Tensor, list, *weave._types]
            if not isinstance(data, cuda_accepted_types):
                raise TypeError(f'Invalid data type for Tensor: {data.__class__.__name__}')
            # By using ellipsis indexing, we can refer to all the dimensions of the Tensor
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
        enable_grad = self._grad_enabled and other.grad_enabled
        out = Tensor(data=(self.data + other.data), _children=(self, other), _op='+',
                     use_grad=self._grad_enabled, device=self.device)

        def _backward():
            # backward pass for addition operations (gradient flows toward the children)
            obj_list = [self, other]
            for obj in obj_list:
                if out.grad.shape != obj.shape:
                    obj.grad += out.grad[1].unsqueeze(0)
                else:
                    obj.grad += out.grad

        if enable_grad:
            out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __radd__(self, other: Any) -> "Tensor":
        return self + other

    def __iadd__(self, other: Any) -> "Tensor":
        enable_grad = self._grad_enabled and other.grad_enabled
        out = self + other
        if enable_grad:
            HookManager.create_hooks(self, '+', alter=True, other=other)
        self._op = out._op
        self.grad = out.grad
        self.data = out.data
        self._populate(out.data)
        return self

    def __sub__(self, other: Any) -> "Tensor":
        return self + (-other)

    def __rsub__(self, other: Any) -> "Tensor":
        return (-self) + other

    def __isub__(self, other: Any) -> "Tensor":
        enable_grad = self._grad_enabled and other.grad_enabled
        out = self + (-other)
        if enable_grad:
            HookManager.create_hooks(self, '-', alter=True, other=other)
        self._op = out._op
        self.grad = out.grad
        self._data = out.data
        self._populate(out.data)
        return self

    def __mul__(self, other: Any) -> "Tensor":
        if type(other) in [np.ndarray, list]:
            other = Tensor(data=other, use_grad=self._grad_enabled, device=self.device)
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
        enable_grad = (self._grad_enabled and other.grad_enabled)
        out = Tensor(data=(self.data * other.data), _children=(self, other), _op='*', use_grad=enable_grad,
                     device=self.device)

        def _backward():
            # backward pass using the chain rule for multiplication
            self._grad_enabled = False
            other._grad_enabled = False

            self.grad += other * out.grad
            other.grad += self * out.grad

            self._grad_enabled = True
            other._grad_enabled = True

        if enable_grad:
            out._backward = _backward
        return out

    def __rmul__(self, other: Any) -> "Tensor":
        return self * other

    def __imul__(self, other: Any) -> "Tensor":
        enable_grad = self._grad_enabled and other.grad_enabled
        out = self * other
        if enable_grad:
            HookManager.create_hooks(self, '*', alter=True, other=other)
        self._op = out._op
        self.grad = out.grad
        self.data = out.data
        self._populate(out.data)
        return self

    def __pow__(self, other: Any) -> "Tensor":
        if type(other) in [np.ndarray, list]:
            other = Tensor(data=other, use_grad=self._grad_enabled, device=self.device)
        elif isinstance(other, (int, float)):
            val = self.data
            val.astype(float)
            out = Tensor(data=(self.data ** other), _children=(self,), _op='**', use_grad=self._grad_enabled,
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
        enable_grad = self._grad_enabled and other.grad_enabled
        out = Tensor(data=(self.data ** other.data), _children=(self, other), _op='**', use_grad=enable_grad,
                     device=self.device)

        def _backward():
            self._grad_enabled = False
            other._grad_enabled = False
            # backward pass making use of the partial derivative of powers
            self.grad += other * self ** (other - 1) * out.grad  # d/dx x**y = y*x**(y - 1)
            other.grad += (self ** other) * self.log() * out.grad  # d/dy x**y = (x**y) * ln(x)
            self._grad_enabled = True
            other._grad_enabled = True

        if enable_grad:
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
        enable_grad = self._grad_enabled and other.grad_enabled
        out = Tensor(data=(self.data / other.data), _children=(self, other), _op='/', use_grad=enable_grad,
                     device=self.device)

        def _backward():
            self._grad_enabled = False
            other._grad_enabled = False
            # We use the partial derivatives for the division of variables
            self.grad += out.grad / other  # d/dx x/y = 1/y
            other.grad += -((self / other ** 2) * out.grad)  # d/dy x/y = x/(y**2)
            self._grad_enabled = True
            other._grad_enabled = True

        if enable_grad:
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
        enable_grad = self._grad_enabled and other.grad_enabled
        out = Tensor(data=(self.data @ other.data), _children=(self, other), _op='@', use_grad=enable_grad,
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
                    if jump:
                        raise ValueError
                    res = out.grad.data @ following.data.T
                    assert res.shape == current.shape, 'Error1'
                    current.grad += Tensor(data=res, device=out.device)
                    jump = True
                except (ValueError, AssertionError):
                    # 2.
                    res = following.data.T @ out.grad.data
                    current.grad += Tensor(data=res, device=out.device)

        if enable_grad:
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
        """
        Calculate the natural logarithm of the tensor elementwise.
        """
        if self.device == 'cpu':
            out = Tensor(data=np.log(self.data), _children=(self,), _op='log', use_grad=self._grad_enabled,
                         device=self.device)
        else:
            out = Tensor(data=cp.log(self.data), _children=(self,), _op='log', use_grad=self._grad_enabled,
                         device=self.device)

        def _backward():
            self.grad += (1 / Tensor(data=self.data, device=self.device)) * out.grad  # d/dx ln(x) = 1/x

        if self._grad_enabled:
            out._backward = _backward
        return out

    def __abs__(self):
        if self.device == 'cpu':
            out = Tensor(data=np.abs(self.data), _children=(self,), _op='abs', use_grad=self._grad_enabled,
                         device=self.device)
        else:
            out = Tensor(data=cp.abs(self.data), _children=(self,), _op='abs', use_grad=self._grad_enabled,
                         device=self.device)

        def _backward():
            self.grad += np.where(self > 0, 1, -1) * out.grad

        if self._grad_enabled:
            out._backward = _backward
        return out

    def __lt__(self, other: Any) -> "Tensor":
        if type(other) in [np.ndarray, list]:
            other = Tensor(data=other, use_grad=self._grad_enabled, device=self.device)
        elif isinstance(other, (int, float)):
            return Tensor(data=self.data < other, dtype=float, use_grad=self._grad_enabled, device=self.device)
        elif not isinstance(other, Tensor):
            raise TypeError(f"Cannot compare 'Tensor' with '{other.__class__.__name__}'")
        return Tensor(data=self.data < other.data, dtype=float, use_grad=self._grad_enabled, device=self.device)

    def __le__(self, other: Any) -> "Tensor":
        if type(other) in [np.ndarray, list]:
            other = Tensor(data=other, use_grad=self._grad_enabled, device=self.device)
        elif isinstance(other, (int, float)):
            return Tensor(data=self.data <= other, dtype=float, use_grad=self._grad_enabled, device=self.device)
        elif not isinstance(other, Tensor):
            raise TypeError(f"Cannot compare 'Tensor' with '{other.__class__.__name__}'")
        return Tensor(data=self.data <= other.data, dtype=float, use_grad=self._grad_enabled, device=self.device)

    def __gt__(self, other: Any) -> "Tensor":
        if type(other) in [np.ndarray, list]:
            other = Tensor(data=other, use_grad=self._grad_enabled, device=self.device)
        elif isinstance(other, (int, float)):
            return Tensor(data=self.data > other, dtype=float, use_grad=self._grad_enabled, device=self.device)
        elif not isinstance(other, Tensor):
            raise TypeError(f"Cannot compare 'Tensor' with '{other.__class__.__name__}'")
        return Tensor(data=self.data > other.data, dtype=float, use_grad=self._grad_enabled, device=self.device)

    def __ge__(self, other: Any) -> "Tensor":
        if type(other) in [np.ndarray, list]:
            other = Tensor(data=other, use_grad=self._grad_enabled, device=self.device)
        elif isinstance(other, (int, float)):
            return Tensor(data=self.data >= other, dtype=float, use_grad=self._grad_enabled, device=self.device)
        elif not isinstance(other, Tensor):
            raise TypeError(f"Cannot compare 'Tensor' with '{other.__class__.__name__}'")
        return Tensor(data=self.data >= other.data, dtype=float, use_grad=self._grad_enabled, device=self.device)

    def __eq__(self, other):
        if type(other) in [np.ndarray, list]:
            other = Tensor(data=other, use_grad=self._grad_enabled, device=self.device)
        elif isinstance(other, (int, float)):
            return Tensor(data=self.data == other, dtype=float, use_grad=self._grad_enabled, device=self.device)
        elif not isinstance(other, Tensor):
            raise TypeError(f"Cannot compare 'Tensor' with '{other.__class__.__name__}'")
        return Tensor(data=self.data == other.data, dtype=float, use_grad=self._grad_enabled, device=self.device)

    def exp(self):
        return np.e ** self

    @property
    def data(self) -> Union[np.ndarray, 'cp.ndarray']:
        # We have to reset the 'data' property since NumPy already makes use of it
        return self._data

    @data.setter
    def data(self, value):
        if self.device == 'cpu':
            if value is None:
                self._data = np.asarray(self)
            else:
                types = Union[np.ndarray, list, *weave._types, int, float, Tensor]
                self._data = value.get() if not isinstance(value, types) else np.asarray(value, dtype=self.dtype)
        else:
            if value is None:
                # cupy tends to make arrays of nan in some cases, so we have to fix for it
                self._data = cp.nan_to_num(cp.asarray(self, dtype=self.dtype))
            else:
                self._data = value if isinstance(value, cp.ndarray) else cp.asarray(value, dtype=self.dtype)

    @property
    def T(self):
        """
        Transposes all dimensions of the tensor.
        """
        return self.transpose()

    @property
    def mT(self):
        """
        Transposes last two dimensions of the tensor.
        """
        if self.ndim >= 2:
            return self._last_transposed()
        raise ValueError(f"Tensor must have two dimensions or more to perform this operation.")

    @property
    def grad_enabled(self) -> bool:
        """
        Returns True if the gradient is enabled for the tensor, False otherwise.
        """
        return self._grad_enabled

    def backward(self):
        """
        Performs the backpropagation algorithm from the tensor is being called on through all the tensors that were
        involved on the computations that resulted in it. Calculates the gradients of all tensors based on the
        operations they performed.
        """
        # Backpropagation algorithm to traverse the operation graph that we have built with each mathematical operation
        topo = []
        visited = set()

        def build_topo(t: "Tensor"):
            # We build a function to perform a topological sort on all the tensors involved in the creation of the
            # tensor that the backward was called on.
            if id(t) not in visited:
                visited.add(id(t))
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)
        if self.shape == ():
            self.grad = Tensor(data=1., device=self.device)
        else:
            self.grad = (Tensor(self.shape, device=self.device) * 0) + 1  # We set the first tensor's gradient to 1
        for node in reversed(topo):
            node._backward()  # Run the backwards function of all tensors in reverse

    def cpu(self):
        """
        Transfers the tensor and all its attributes to the CPU memory if not already in it.
        """
        if not (self.device == 'cpu'):
            self.device = weave.cuda.Device('cpu')
            self.data = self.data.get()

    def cuda(self):
        """
        Transfers the tensor and all its attributes to the GPU memory if not already in it.
        """
        if self.device == 'cpu':
            self.device = weave.cuda.Device('cuda')
            self.data = cp.asarray(self.data)

    # From this point onward, we implement many of the methods that a subclass from numpy.ndarray is expected to have.
    # Due to the sheer amount of methods, however, only some of them have been implemented, and these have been selected
    # according to how common we believe them to be in a Machine Learning environment.
    # Another thing to note is that some of the arguments or keyword arguments for the methods are left unused even when
    # they are still defined. This is to maintain compatibility with CuPy's array model as well as sticking to NumPy's
    # method naming conventions.

    def abs(self):
        return abs(self)

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        """
        Returns a copy of the tensor with its dtype changed to that of the input.
        """
        self.data = self.data.astype(dtype, order, copy=copy)

    def all(self, axis=None, out=None, keepdims=False, *, where=True):
        """
        Returns True if all elements evaluate to True.
        """
        if self.device == 'cpu':
            return self.data.all()
        raise NotImplementedError('Cannot perform this operation on tensors on the GPU.')

    def any(self, axis=None, out=None, keepdims=False, *, where=True):
        """
        Returns True if any of the elements evaluate to True.
        """
        if self.device == 'cpu':
            return self.data.any()
        raise NotImplementedError('Cannot perform this operation on tensors on the GPU.')

    def sort(self, axis=-1, kind=None, order=None):
        """
        Sort the elements of the tensor in place.
        """
        self.data = self.data.sort(axis, kind, order)

    def argmax(self, axis=None, out=None, *, keepdims=False):
        """
        Returns the indices of the maximum values along a given axis.
        """
        return Tensor(data=self.data.argmax(), dtype=int, device=self.device)

    def argmin(self, axis=None, out=None, *, keepdims=False):
        """
        Returns the indices of the minimum values along a given axis.
        """
        return Tensor(data=self.data.argmin(), dtype=int, device=self.device)

    def argsort(self, axis=-1, kind=None, order=None):
        """
        Returns the indices that would sort this tensor.
        """
        return Tensor(data=self.data.argsort(), device=self.device)

    def argpartition(self, kth, axis=-1, kind=None, order=None):
        """
        Returns the indices that would partition this tensor. Currently only supported for tensors in the CPU.
        """
        if self.device == 'cpu':
            return Tensor(data=self.data.argpartition(kth), device=self.device)
        raise NotImplementedError('Cannot perform this operation on tensors on the GPU.')

    def byteswap(self, inplace=...):
        """
        Swap the bytes of the tensor elements, switching between small and big endian. Currently only supported for
        tensors in the CPU.
        """
        if self.device == 'cpu':
            return Tensor(data=self.data.byteswap(), device=self.device)
        raise NotImplementedError('Cannot perform this operation on tensors on the GPU.')

    def choose(self, choices, out=..., mode=...):
        raise NotImplementedError('This operation is not implemented for the Tensor class.')

    def clip(self, min: int | None = None, max: int | None = None, out=None, **kwargs):
        """
        Returns a tensor with its values limited to those in the range [min, max].
        """
        return Tensor(data=self.data.clip(min, max, out, **kwargs), device=self.device)

    def compress(self, a, axis=None, out=None):
        """
        Return the selected slices if the tensor along a given axis.
        """
        return Tensor(data=self.data.compress(a, axis, out), device=self.device)

    def conjugate(self):
        """
        Return the elementwise conjugate of the tensor.
        """
        return Tensor(data=self.data.conjugate(), device=self.device)

    def conj(self):
        """
        Alias for Tensor.conjugate().
        """
        return self.conjugate()

    def copy(self, order='C'):
        """
        Returns a copy of the array with all its elements.
        """
        if hasattr(self, '_prev'):
            return Tensor(data=self.data.copy(order), _children=self._prev, _op=self._op, dtype=self.dtype,
                          use_grad=self._grad_enabled, device=self.device)
        else:
            return Tensor(data=self.data.copy(order), _op=self._op, dtype=self.dtype, use_grad=self._grad_enabled,
                          device=self.device)

    def cumprod(self, axis=None, dtype=None, out=None):
        """
        Returns the cumulative product of the elements of the tensor along a given axis.
        """
        return Tensor(data=self.data.cumprod(axis, dtype, out), dtype=self.dtype, device=self.device)

    def cumsum(self, axis=None, dtype=None, out=None):
        """
        Returns the cumulative sum of the elements of the tensor along a given axis.
        """
        return Tensor(data=self.data.cumsum(axis, dtype, out), dtype=self.dtype, device=self.device)

    def diagonal(self, offset=0, axis1=0, axis2=1):
        """
        Returns the diagonal of the elements of the tensor by a given offset and axis.
        """
        return Tensor(data=self.data.diagonal(offset, axis1, axis2), dtype=self.dtype, device=self.device)

    def diag(self, offset=0, axis1=0, axis2=1):
        """
        Alias of Tensor.diagonal()
        """
        return self.diagonal(offset, axis1, axis2)

    def dot(self, b: "Tensor", out=None):
        """
        Returns the dot product of two tensors.
        """
        return Tensor(data=self.data.dot(b.data), use_grad=self._grad_enabled, device=self.device)

    def fill(self, value):
        """
        Fills the tensor with a provided scalar value.
        """
        new = self.data
        new.fill(value)
        return Tensor(data=new, use_grad=self._grad_enabled, device=self.device)

    def flatten(self, start_dim: int = 0, end_dim: int = -1, order='C') -> "Tensor":
        """
        Flattens the tensor starting at `start_dim` and ending at `end_dim'. If no arguments are provided it will
        flatten the tensor along all dimensions.
        """
        shape = []
        for i in self.shape[:start_dim]:
            shape.append(i)
        flattened_dim = np.prod(self.shape[start_dim:end_dim + 1])
        shape.append(flattened_dim)
        for i in self.shape[end_dim + 1: self.ndim]:
            shape.append(i)
        print(shape)
        out = self.reshape(shape)
        return out

    def reshape(self, shape, /, *, order='C'):
        """
        Returns a tensor with the given shape.
        """
        out = Tensor(data=self.data.reshape(shape, order=order), _children=(self,), _op='reshape',
                     use_grad=self._grad_enabled, device=self.device)

        def _backward():
            self.grad += out.grad.reshape(self.shape)

        if self._grad_enabled:
            out._backward = _backward
        return out

    def getfield(self, dtype, offset=0):
        """
        Returns a field view fo the real or complex parts of a tensor given its offset. Currently only supported for
        tensors in the CPU.
        """
        if self.device == 'cpu':
            return Tensor(data=self.data.getfield(dtype, offset), use_grad=self._grad_enabled, device=self.device)
        raise NotImplementedError('This operation is not implemented for the Tensor class.')

    def itemset(self, *args):
        """
        Insert a scalar into the tensor at the specified location.
        """
        if self.device == 'cpu':
            val = self.data
            val.itemset(*args)
            return Tensor(data=val, use_grad=self._grad_enabled, device=self.device)
        else:
            idx = [*([0] * (len(self.shape) - 1)), args[0]] if isinstance(args[0], int) else args[0]
            val = self.data
            val[*idx] = args[1]
            return Tensor(data=val, use_grad=self._grad_enabled, device=self.device)

    def max(self, axis=None, out=None, keepdims=False, initial=None, where=True):
        """
        Return the maximum value along a given axis.
        """
        return Tensor(data=self.data.max(axis, out, keepdims), use_grad=self._grad_enabled,
                      device=self.device)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
        """
        Returns the average of the tensor elements along a given axis.
        """
        out = Tensor(data=self.data.mean(axis, dtype, out, keepdims), _children=(self,), _op='mean',
                     use_grad=self._grad_enabled, device=self.device)

        if axis is None:
            def _backward():
                val = Tensor(data=np.full_like(np.random.rand(*self.shape), (1 / self.size)), device=self.device)
                if len(val.shape) < 2:
                    val = val.unsqueeze(1)
                self.grad += val * out.grad
        else:
            def _backward():
                s = list(self.shape)
                s.remove(self.shape.index(axis))
                fill_value = np.full_like(np.random.rand(*s), 1 / np.prod(s))
                self.grad += Tensor(data=np.array([fill_value for _ in range(self.shape.index(axis))]),
                                    device=self.device) * out.grad

        if self._grad_enabled:
            out._backward = _backward
        return out

    def min(self, axis=None, out=None, keepdims=False, initial=None, where=True):
        """
        Returns the minimum value along a given axis.
        """
        return Tensor(data=self.data.min(axis, out, keepdims), use_grad=self._grad_enabled, device=self.device)

    def nonzero(self):
        """
        Returns the indices of all non-zero elements of the tensor.
        """
        non_zero_tuple = self.data.nonzero()
        return tuple(map(lambda x: Tensor(data=x, use_grad=self._grad_enabled, device=self.device), non_zero_tuple))

    def prod(self, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
        """
        Returns the product of the elements of the tensor along a given axis.
        """
        return Tensor(data=self.data.prod(axis, dtype, out, keepdims), use_grad=self._grad_enabled, device=self.device)

    def ptp(self, axis=None, out=None, keepdims=False):
        """
        Returns the difference between the minimum and maximum elements of the tensor along a given axis.
        """
        return Tensor(data=self.data.ptp(axis, out, keepdims), use_grad=self._grad_enabled, device=self.device)

    def put(self, ind, v, mode='raise'):
        """
        Sets the elements of the tensor to those provided in the inputted locations.
        """
        val = self.data
        val.put(ind, v, mode=mode)
        self.data = val

    def ravel(self, order='C'):
        """
        Returns a flattened array.
        """
        return Tensor(data=self.data.ravel(order), use_grad=self._grad_enabled, device=self.device)

    def repeat(self, repeats, axis=None):
        """
        Repeats the elements of the tensor a 'repeats' number of times along a given axis.
        """
        return Tensor(data=self.data.repeat(repeats, axis), use_grad=self._grad_enabled, device=self.device)

    def round(self, decimals=0, out=None):
        """
        Rounds all the elements of the tensor to a given number of decimals.
        """
        return Tensor(data=self.data.round(decimals, out), use_grad=self._grad_enabled, device=self.device)

    def squeeze(self, axis=0) -> "Tensor":
        """
        Removes axis of length one from the tensor. Default of 0.
        """
        try:
            out = Tensor(data=self.data.squeeze(axis), use_grad=self._grad_enabled, device=self.device)
            if not isinstance(self.grad, int):
                out.grad = Tensor(data=self.grad.data.squeeze(axis), device=self.device)
        except ValueError:
            out = self
        return out

    def unsqueeze(self, axis=0) -> "Tensor":
        """
        Adds axis of length 1 to the tensor in the given axis. Default of 0.
        """
        val = np.expand_dims(self.data, axis=axis) if isinstance(self.data, np.ndarray) else \
              cp.expand_dims(self.data, axis=axis)
        out = Tensor(data=val, use_grad=self._grad_enabled, device=self.device)
        if not isinstance(self.grad, int):
            val = np.expand_dims(self.grad.data, axis=axis) if isinstance(self.grad, np.ndarray) else \
                  cp.expand_dims(self.grad.data, axis=axis)
            out.grad = Tensor(data=val, device=self.device)
        return out

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True):
        """
        Computes the standard deviation of the elements of the tensor along a given axis.
        """
        val = self.data.std(axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)
        return Tensor(data=val, use_grad=self._grad_enabled, device=self.device)

    def sum(self, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
        """
        Returns the sum of the elements of the tensor along a given axis. If no axis is provided, it adds all elements.
        """
        val = self.data.sum(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
        out = Tensor(data=val, _children=(self,), _op='sum', use_grad=self._grad_enabled, device=self.device)

        if axis is None:
            def _backward():
                self.grad += Tensor(data=np.ones(self.shape), device=self.device) * out.grad
        else:
            def _backward():
                out_grad = out.grad.data if isinstance(out.grad.data, np.ndarray) else out.grad.data.get()
                s = list(self.shape)
                s[axis] = -1
                arr = np.array(out_grad).reshape(*s)
                self.grad += Tensor(data=(np.repeat(arr, self.shape[axis], axis=axis)))

        if self._grad_enabled:
            out._backward = _backward
        return out

    def take(self, indices, axis=None, out=None, mode='raise'):
        """
        Extracts the elements of the tensor on the given indices.
        """
        return Tensor(data=self.data.take(indices, axis=axis, out=out), use_grad=self._grad_enabled,
                      device=self.device)

    def tolist(self) -> list:
        """
        Returns the elements of the tensor as a list.
        """
        return self.data.tolist()

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        """
        Return the sum along the diagonal of the tensor.
        """
        val = self.data.trace(offset, axis1, axis2, dtype, out=out)
        return Tensor(data=val, use_grad=self._grad_enabled, device=self.device)

    def transpose(self, *axes):
        """
        Transposes the elements of the tensor along the given axes. By default, it transposes all dimensions of the tensor.
        """
        out = Tensor(data=self.data.transpose(*axes), _children=(self,), _op='T', use_grad=self._grad_enabled,
                     device=self.device)

        def _backward():
            self.grad += out.grad.transpose()

        if self._grad_enabled:
            out._backward = _backward
        return out

    def _last_transposed(self):
        if self.device == 'cpu':
            array = np.einsum('...ij->...ji', self.data)
        else:
            array = cp.einsum('...ij->...ji', self.data)
        out = Tensor(data=array, use_grad=self._grad_enabled, device=self.device)

        def _backward():
            if self.device == 'cpu':
                ans = np.einsum('...ij->...ji', out.grad)
            else:
                ans = cp.einsum('...ij->...ji', out.grad)
            self.grad += ans

        if self._grad_enabled:
            out._backward = _backward
        return out

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True):
        """
        Returns the variance of the elements of the tensor along a given axis.
        """
        val = self.data.var(axis, dtype, out, ddof, keepdims)
        return Tensor(data=val, use_grad=self._grad_enabled, device=self.device)

    def __getitem__(self, idx: int | slice | tuple):
        if isinstance(idx, (int, slice)):
            return Tensor(data=self.data[idx], use_grad=self._grad_enabled, device=self.device)
        elif isinstance(idx, tuple):
            try:  # manage in case we are indexing a number
                r = float(self.data[idx])
            except TypeError:
                r = self.data[idx]
            return Tensor(data=r)
        elif isinstance(idx, (list, np.ndarray)):
            return Tensor(data=self.data[tuple(idx)], use_grad=self._grad_enabled, device=self.device)
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def view(self):
        """
        Returns a view of the tensor, sharing the same data.
        """
        out = Tensor(data=self.data.view(), use_grad=self._grad_enabled, device=self.device)
        out._prev = self._prev
        out._op = self._op
        out.grad = self.grad
        return out

    # Could have implemented more of numpy.ndarray methods, as well as raised a NotImplementedError on those who aren't,
    # but I'm so done with this class that I'll leave it like this and perhaps come back to do it at some point later.

    def __str__(self):
        info = self.data.get() if not isinstance(self.data, np.ndarray) else self.data
        data_string = np.asarray(info, dtype=self.dtype).__repr__()
        data_string = data_string[6:-1].rsplit('\n') if 'array' in data_string else data_string.rsplit('\n')
        data_string = [data_string[0]] + [' ' + line.strip() for line in data_string[1:]]
        data_string = '\n'.join(data_string)
        data_string = re.sub(r', dtype=\w+$', '', data_string)
        return data_string

    def __repr__(self):
        # Display the Tensors in a way that is consistent to how NumPy, PyTorch, Tensorflow and the such do.
        info = self.data.get() if not isinstance(self.data, np.ndarray) else self.data
        data_string = np.asarray(info, dtype=self.dtype).__repr__()
        data_string = re.sub(r',\s\[', ',\n       [', data_string)
        data_string = data_string[6:-1].rsplit('\n') if 'array' in data_string else data_string.rsplit('\n')
        data_string = [data_string[0]] + [' ' + line for line in data_string[1:]]
        data_string = '\n'.join(data_string)
        data_string = re.sub(r', dtype=\w+$', '', data_string)
        s = f'Tensor({data_string}, dtype={self.dtype}' if not self._grad_enabled else f'Tensor({data_string}, ' \
                                                                                       f'dtype={self.dtype}, ' \
                                                                                       f'uses_grad={self._grad_enabled}'
        if self.device != 'cpu':
            s += f', device={self.device}'
        s += ')'
        return s
