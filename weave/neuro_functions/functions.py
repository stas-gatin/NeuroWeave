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

Please note that this is an extension for the Numpy library.
For detailed information, refer to the official documentation
at https://numpy.org/doc/.
"""

import numpy as np

from weave import Tensor

__all__ = [
    'ones',
    'ones_like',
    'empty',
    'zeros',
    'zeros_like',
    'full',
    'full_like',
    'rand',
    'eye',
    'linspace',
    'arange',
    'dot',
    'matmul',
    'inner',
    'outer',
    'tensordot',
    'einsum',
    'trace',
    'norm',
    'inv',
    'tensorinv',
    'all',
    'any',
    'tensor',
    'diag',
    'tril',
    'triu',
    'concatenate',
    'stack',
    'vstack',
    'hstack',
    'dstack',
    'split',
    'dsplit',
    'vsplit',
    'hsplit',
    'delete',
    'append',
    'resize',
    'load',
    'save'
]


def ones(shape: int | tuple, dtype: str = None,
         use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Return a new array of given shape and type, filled with ones."""

    array = np.ones(shape, dtype=dtype)
    print(type(shape))
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def ones_like(tensor_like: Tensor, shape: int | tuple, dtype: str = None,
              use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Return an array of ones with the same shape and type as a given array."""

    array = np.ones_like(a=tensor_like.data, dtype=dtype, shape=shape)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def empty(shape: int | tuple, dtype: str = None,
          use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Return a new array of given shape and type, without initializing entries."""

    array = np.empty(shape, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def full(shape: int | tuple, fill_value=np.inf, dtype: str = None,
         use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Return a new array of given shape and type, filled with fill_value."""

    array = np.full(shape, fill_value=fill_value, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def full_like(tensor_like: Tensor, shape: int | tuple, fill_value=np.inf,
              dtype: str = None,
              use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Return a full array with the same shape and type as a given array."""

    array = np.full_like(a=tensor_like.data, fill_value=fill_value, dtype=dtype,
                         shape=shape)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def zeros(shape: int | tuple, dtype: str = None,
          use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Return a new array of given shape and type, filled with zeros."""

    array = np.zeros(shape, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def zeros_like(tensor_like: Tensor, shape: int | tuple, dtype: str = None,
               use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Return an array of zeros with the same shape and type as a given array."""

    array = np.zeros_like(a=tensor_like.data, dtype=dtype, shape=shape)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def rand(*shape: int | tuple,
         use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Random values in a given shape."""

    array = np.random.rand(*shape)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def eye(rows: int = None, columns: int = None, k: int = 0, dtype: str = None,
        use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Return a 2-D array with ones on the diagonal and zeros elsewhere."""

    array = np.eye(rows, columns, k, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def linspace(start, stop, num, dtype: str = None,
             use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """
    Return evenly spaced numbers over a specified interval.

    Returns num evenly spaced samples, calculated over the interval [start, stop].

    The endpoint of the interval can optionally be excluded.
    """

    array = np.linspace(start, stop, num, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def arange(start_, stop_, step, dtype: str = None,
           use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """
    Return evenly spaced values within a given interval.

    arange can be called with a varying number of positional arguments:

    arange(stop): Values are generated within the half-open interval [0, stop) (in other words, the interval including start but excluding stop).
    arange(start, stop): Values are generated within the half-open interval [start, stop).
    arange(start, stop, step) Values are generated within the half-open interval [start, stop), with spacing between values given by step.
    """

    array = np.arange(start_, stop_, step, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def dot(a, b, dtype: str = None, use_grad: bool = False,
        device: str = 'cpu') -> Tensor:
    """Dot product of two arrays."""

    array = np.dot(a, b)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def matmul(a, b, use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Matrix product of two arrays."""
    array = np.matmul(a, b)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def inner(a, b, use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Inner product of two arrays."""
    array = np.inner(a, b)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def outer(a, b, use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Compute the outer product of two vectors."""
    array = np.outer(a, b)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def tensordot(a, b, use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Compute tensor dot product along specified axes for arrays >= 1-D."""
    array = np.tensordot(a, b)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def einsum(subscripts, *operands, use_grad: bool = False,
           device: str = 'cpu') -> Tensor:
    """Evaluates the Einstein summation convention on the operands."""
    array = np.einsum(subscripts, *operands)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None,
          use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Return the sum along diagonals of the array."""
    array = np.trace(a, offset, axis1, axis2, dtype, out)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def norm(x, ord=None, axis=None, keepdims=False, use_grad: bool = False,
         device: str = 'cpu') -> Tensor:
    """Matrix or vector norm."""
    array = np.linalg.norm(x, ord, axis, keepdims)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def inv(a, use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Compute the (multiplicative) inverse of a matrix."""
    array = np.linalg.inv(a)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def tensorinv(a, ind=2, use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Compute the 'inverse' of an N-dimensional array."""
    array = np.linalg.tensorinv(a, ind)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def all(x, axis=None, out=None, keepdims=False, use_grad: bool = False,
        device: str = 'cpu') -> Tensor:
    """Test whether all array elements along a given axis evaluate to True."""
    array = np.all(x, axis, out, keepdims)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def any(x, axis=None, out=None, keepdims=False, use_grad: bool = False,
        device: str = 'cpu') -> Tensor:
    """Test whether any array element along a given axis evaluates to True."""
    array = np.any(x, axis, out, keepdims)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def tensor(data: list, dtype: str = None, use_grad: bool = False,
           device: str = 'cpu') -> Tensor:
    """Create an tensor from a data list."""

    return Tensor(data=data, dtype=dtype, use_grad=use_grad)


def diag(tensor_like: Tensor, k: int = 0,
         use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Extract a diagonal or construct a diagonal array."""

    array = np.diag(tensor_like.data, k)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def tril(tensor_like: Tensor, k: int = 0,
         use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Lower triangle of an array."""

    array = np.tril(tensor_like.data, k)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def triu(tensor_like: Tensor, k: int = 0,
         use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """
    Upper triangle of an array.
    """
    array = np.triu(tensor_like.data, k)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def concatenate(tensors: tuple = None, axis: int = 0, dtype: str = None,
                use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Join a sequence of arrays along an existing axis."""
    tensors_data = []
    for tensor_data in tensors:
        tensors_data.append(tensor_data.data)
    array = np.concatenate(tensors_data, axis=axis, dtype=dtype)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def stack(tensors: tuple = None, axis: int = 0, dtype: str = None,
          use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Join a sequence of arrays along a new axis."""

    tensors_data = []
    for tensor_data in tensors:
        tensors_data.append(tensor_data.data)
    array = np.stack(tensors_data, axis=axis, dtype=dtype)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def vstack(tensors: tuple = None, dtype: str = None,
           use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Stack arrays in sequence vertically (row wise)."""

    tensors_data = []
    for tensor_data in tensors:
        tensors_data.append(tensor_data.data)
    array = np.vstack(tup=tensors_data, dtype=dtype)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def hstack(tensors: tuple = None, dtype: str = None,
           use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Stack arrays in sequence horizontally (column wise)."""

    tensors_data = []
    for tensor_data in tensors:
        tensors_data.append(tensor_data.data)
    array = np.hstack(tup=tensors_data, dtype=dtype)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def dstack(tensors: tuple = None,
           use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Stack arrays in sequence depth wise (along third axis)."""

    tensors_data = []
    for tensor_data in tensors:
        tensors_data.append(tensor_data.data)
    array = np.dstack(tup=tensors_data)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def split(ten: Tensor, indices_or_sections: int, axis: int = 0,
          use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Split an array into multiple sub-arrays as views into ary."""

    array = np.split(ary=np.array(ten.data),
                     indices_or_sections=indices_or_sections, axis=axis)
    return Tensor(data=array, use_grad=use_grad)


def dsplit(ten: Tensor, indices_or_sections: int, axis: int = 0,
           use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Split array into multiple sub-arrays along the 3rd axis (depth)."""

    array = np.dsplit(ary=np.array(ten.data),
                      indices_or_sections=indices_or_sections)
    return Tensor(data=array, use_grad=use_grad)


def vsplit(ten: Tensor, indices_or_sections: int, axis: int = 0,
           use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Split an array into multiple sub-arrays vertically (row-wise)."""

    array = np.vsplit(ary=np.array(ten.data),
                      indices_or_sections=indices_or_sections)
    return Tensor(data=array, use_grad=use_grad)


def hsplit(ten: Tensor, indices_or_sections: int, axis: int = 0,
           use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Split an array into multiple sub-arrays horizontally (column-wise)."""

    array = np.hsplit(ary=np.array(ten.data),
                      indices_or_sections=indices_or_sections)
    return Tensor(data=array, use_grad=use_grad)


def delete(ten: Tensor, obj: slice | int, axis: int = 0,
           use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """
    Return a new array with sub-arrays along an axis deleted.
    For a one dimensional array, this returns those entries
    not returned by arr[obj].
    """

    array = np.delete(arr=ten.data, obj=obj, axis=axis)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def append(ten: Tensor, values: Tensor, axis: int = None,
           use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Append values to the end of an array."""

    array = np.append(arr=ten.data, values=values, axis=axis)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def resize(ten: Tensor, new_shape: int | tuple,
           use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Return a new array with the specified shape.

    If the new array is larger than the original array,
    then the new array is filled with repeated copies of ten.
    Note that this behavior is different from a.resize(new_shape)
    which fills with zeros instead of repeated copies of ten.
    """

    array = np.resize(a=ten.data, new_shape=new_shape)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def unique(ten: Tensor, axis: int = None,
           use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Find the unique elements of an array."""

    array = np.unique(ar=ten.data, axis=axis)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def flip(ten: Tensor, axis: int = None,
         use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Reverse the order of elements in an array along the given axis.

    The shape of the array is preserved, but the elements are reordered."""

    array = np.flip(m=ten.data, axis=axis)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def reshape(ten: Tensor, new_shape: int | tuple = None,
            use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Gives a new shape to an array without changing its data."""

    array = np.reshape(a=ten.data, newshape=new_shape)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def rot90(ten: Tensor, k: int = None, axes=(0, 1),
          use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Rotate an array by 90 degrees in the plane specified by axes."""

    array = np.rot90(m=ten.data, k=k, axes=axes)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def block(tensors: Tensor,
          use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Assemble an nd-array from nested lists of blocks."""

    tensors_data = []
    for tensor_data in tensors:
        tensors_data.append(tensor_data.data)
    array = np.block(arrays=tensors_data)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def load(filename: str, use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Load a tensor from a file."""
    array = np.load(filename)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def save(filename: str, ten: Tensor, use_grad: bool = False,
         device: str = 'cpu') -> None:
    """Save a tensor to a file."""
    np.save(filename, ten.data)
