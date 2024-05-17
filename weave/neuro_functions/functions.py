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

import weave.cuda

if weave.cuda.is_available():
    import cupy as cp

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
    'randn',
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
    'save',
    'exp',
    'log',
    'abs',
    'sin',
    'cos',
    'tan',
    'sinh',
    'cosh',
    'tanh',
    'prod',
    'sum',
    'cumprod',
    'cumsum',
    'max',
    'min',
    'convolve',
    'sign',
]


def ones(shape: int | tuple, dtype: str = None,
         use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Return a new array of given shape and type, filled with ones."""

    array = np.ones(shape, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad, device=device)


def ones_like(tensor_like: Tensor, shape: int | tuple, dtype: str = None,
              use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Return an array of ones with the same shape and type as a given array."""

    if tensor_like.device == 'cpu':
        array = np.ones_like(a=tensor_like.data, dtype=dtype, shape=shape)
    else:
        array = cp.ones_like(a=tensor_like.data, dtype=dtype, shape=shape)

    return Tensor(data=array, dtype=dtype, use_grad=use_grad, device=device)


def empty(shape: int | tuple, dtype: str = None,
          use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Return a new array of given shape and type, without initializing entries."""

    array = np.empty(shape, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad, device=device)


def full(shape: int | tuple, fill_value=np.inf, dtype: str = None,
         use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Return a new array of given shape and type, filled with fill_value."""

    array = np.full(shape, fill_value=fill_value, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad, device=device)


def full_like(tensor_like: Tensor, shape: int | tuple, fill_value=np.inf,
              dtype: str = None,
              use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Return a full array with the same shape and type as a given array."""

    if tensor_like.device == 'cpu':
        array = np.full_like(a=tensor_like.data, fill_value=fill_value, dtype=dtype,shape=shape)
    else:
        array = cp.full_like(a=tensor_like.data, fill_value=fill_value, dtype=dtype,
                             shape=shape)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad, device=device)


def zeros(shape: int | tuple, dtype: str = None,
          use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Return a new array of given shape and type, filled with zeros."""

    array = np.zeros(shape, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad, device=device)


def zeros_like(tensor_like: Tensor, shape: int | tuple, dtype: str = None,
               use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Return an array of zeros with the same shape and type as a given array."""

    if tensor_like.device == 'cpu':
        array = np.zeros_like(a=tensor_like.data, dtype=dtype, shape=shape)
    else:
        array = cp.zeros_like(a=tensor_like.data, dtype=dtype, shape=shape)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad, device=device)


def rand(*shape: int | tuple,
         use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Random values in a given shape."""

    array = np.random.rand(*shape)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)

def randn(*shape: int | tuple,
          use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Random values in a given shape."""

    array = np.random.randn(*shape)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def eye(rows: int = None, columns: int = None, k: int = 0, dtype: str = None,
        use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Return a 2-D array with ones on the diagonal and zeros elsewhere."""

    array = np.eye(rows, columns, k, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad, device=device)


def linspace(start, stop, num, dtype: str = None,
             use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """
    Return evenly spaced numbers over a specified interval.

    Returns num evenly spaced samples, calculated over the interval [start, stop].

    The endpoint of the interval can optionally be excluded.
    """

    array = np.linspace(start, stop, num, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad, device=device)


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
    return Tensor(data=array, dtype=dtype, use_grad=use_grad, device=device)


def dot(a, b, dtype: str = None, use_grad: bool = False,
        device: str = 'cpu') -> Tensor:
    """Dot product of two arrays."""

    array = np.dot(a, b)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def matmul(a, b, use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Matrix product of two arrays."""
    array = np.matmul(a, b)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def inner(a, b, use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Inner product of two arrays."""
    array = np.inner(a, b)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def outer(a, b, use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Compute the outer product of two vectors."""
    array = np.outer(a, b)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def tensordot(a, b, use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Compute tensor dot product along specified axes for arrays >= 1-D."""
    array = np.tensordot(a, b)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def einsum(subscripts, *operands, use_grad: bool = False,
           device: str = 'cpu') -> Tensor:
    """Evaluates the Einstein summation convention on the operands."""
    array = np.einsum(subscripts, *operands)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None,
          use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Return the sum along diagonals of the array."""
    array = np.trace(a, offset, axis1, axis2, dtype, out)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def norm(x, ord=None, axis=None, keepdims=False, use_grad: bool = False,
         device: str = 'cpu') -> Tensor:
    """Matrix or vector norm."""
    array = np.linalg.norm(x, ord, axis, keepdims)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def inv(a, use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Compute the (multiplicative) inverse of a matrix."""
    array = np.linalg.inv(a)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def tensorinv(a, ind=2, use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Compute the 'inverse' of an N-dimensional array."""
    array = np.linalg.tensorinv(a, ind)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def all(x, axis=None, out=None, keepdims=False, use_grad: bool = False,
        device: str = 'cpu') -> Tensor:
    """Test whether all array elements along a given axis evaluate to True."""
    array = np.all(x, axis, out, keepdims)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def any(x, axis=None, out=None, keepdims=False, use_grad: bool = False,
        device: str = 'cpu') -> Tensor:
    """Test whether any array element along a given axis evaluates to True."""
    array = np.any(x, axis, out, keepdims)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def tensor(data: list, dtype: str = None, use_grad: bool = False,
           device: str = 'cpu') -> Tensor:
    """Create an tensor from a data list."""

    return Tensor(data=data, dtype=dtype, use_grad=use_grad, device=device)


def diag(tensor_like: Tensor, k: int = 0,
         use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Extract a diagonal or construct a diagonal array."""

    if tensor_like.device == 'cpu':
        array = np.diag(tensor_like.data, k)
    else:
        array = cp.diag(tensor_like.data, k)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def tril(tensor_like: Tensor, k: int = 0,
         use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Lower triangle of an array."""

    if tensor_like.device == 'cpu':
        array = np.tril(tensor_like.data, k)
    else:
        array = cp.tril(tensor_like.data, k)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def triu(tensor_like: Tensor, k: int = 0,
         use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Upper triangle of an array."""

    if tensor_like.device == 'cpu':
        array = np.triu(tensor_like.data, k)
    else:
        array = cp.triu(tensor_like.data, k)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def concatenate(tensors: tuple = None, axis: int = 0, dtype: str = None,
                use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Join a sequence of arrays along an existing axis."""

    tensors_data = []
    for tensor_data in tensors:
        tensors_data.append(tensor_data.data)
    array = np.concatenate(tensors_data, axis=axis, dtype=dtype)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def stack(tensors: tuple = None, axis: int = 0, dtype: str = None,
          use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Join a sequence of arrays along a new axis."""

    tensors_data = []
    for tensor_data in tensors:
        tensors_data.append(tensor_data.data)
    array = np.stack(tensors_data, axis=axis, dtype=dtype)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def vstack(tensors: tuple = None, dtype: str = None,
           use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Stack arrays in sequence vertically (row wise)."""

    tensors_data = []
    for tensor_data in tensors:
        tensors_data.append(tensor_data.data)
    array = np.vstack(tup=tensors_data, dtype=dtype)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def hstack(tensors: tuple = None, dtype: str = None,
           use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Stack arrays in sequence horizontally (column wise)."""

    tensors_data = []
    for tensor_data in tensors:
        tensors_data.append(tensor_data.data)
    array = np.hstack(tup=tensors_data, dtype=dtype)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def dstack(tensors: tuple = None,use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Stack arrays in sequence depth wise (along third axis)."""

    tensors_data = []
    for tensor_data in tensors:
        tensors_data.append(tensor_data.data)
    array = np.dstack(tup=tensors_data)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def split(ten: Tensor, indices_or_sections: int, axis: int = 0,
          use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Split an array into multiple sub-arrays as views into ary."""
    if ten.device == 'cpu':
        array = np.split(ary=np.array(ten.data),indices_or_sections=indices_or_sections, axis=axis)
    else:
        array = cp.split(ary=np.array(ten.data),indices_or_sections=indices_or_sections, axis=axis)

    return Tensor(data=array, use_grad=use_grad, device=device)


def dsplit(ten: Tensor, indices_or_sections: int, axis: int = 0,
           use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Split array into multiple sub-arrays along the 3rd axis (depth)."""
    if ten.device == 'cpu':
        array = np.dsplit(ary=np.array(ten.data),indices_or_sections=indices_or_sections)

    else:
        array = cp.dsplit(ary=np.array(ten.data),indices_or_sections=indices_or_sections)

    return Tensor(data=array, use_grad=use_grad, device=device)


def vsplit(ten: Tensor, indices_or_sections: int, axis: int = 0,
           use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Split an array into multiple sub-arrays vertically (row-wise)."""

    if ten.device == 'cpu':
        array = np.vsplit(ary=np.array(ten.data),indices_or_sections=indices_or_sections)
    else:
        array = cp.vsplit(ary=np.array(ten.data),indices_or_sections=indices_or_sections)

    return Tensor(data=array, use_grad=use_grad, device=device)


def hsplit(ten: Tensor, indices_or_sections: int, axis: int = 0,
           use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Split an array into multiple sub-arrays horizontally (column-wise)."""

    if ten.device == 'cpu':
        array = np.hsplit(ary=np.array(ten.data),indices_or_sections=indices_or_sections)
    else:
        array = cp.hsplit(ary=np.array(ten.data),indices_or_sections=indices_or_sections)

        return Tensor(data=array, use_grad=use_grad, device=device)


def delete(ten: Tensor, obj: slice | int, axis: int = 0,
           use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """
    Return a new array with sub-arrays along an axis deleted.
    For a one dimensional array, this returns those entries
    not returned by arr[obj].
    """

    if ten.device == 'cpu':
        array = np.delete(arr=ten.data, obj=obj, axis=axis)
    else:
        array = cp.delete(arr=ten.data, obj=obj, axis=axis)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def append(ten: Tensor, values: Tensor, axis: int = None,
           use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Append values to the end of an array."""

    if ten.device == 'cpu':
        array = np.append(arr=ten.data, values=values, axis=axis)
    else:
        array = cp.append(arr=ten.data, values=values, axis=axis)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def resize(ten: Tensor, new_shape: int | tuple,
           use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Return a new array with the specified shape.

    If the new array is larger than the original array,
    then the new array is filled with repeated copies of ten.
    Note that this behavior is different from a.resize(new_shape)
    which fills with zeros instead of repeated copies of ten.
    """

    if ten.device == 'cpu':
        array = np.resize(a=ten.data, new_shape=new_shape)
    else:
        array = cp.resize(a=ten.data, new_shape=new_shape)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def unique(ten: Tensor, axis: int = None,
           use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Find the unique elements of an array."""

    if ten.device == 'cpu':
        array = np.unique(ar=ten.data, axis=axis)
    else:
        array = cp.unique(ar=ten.data, axis=axis)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def flip(ten: Tensor, axis: int = None,
         use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Reverse the order of elements in an array along the given axis.

    The shape of the array is preserved, but the elements are reordered."""

    if ten.device == 'cpu':
        array = np.flip(m=ten.data, axis=axis)
    else:
        array = cp.flip(m=ten.data, axis=axis)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def reshape(ten: Tensor, new_shape: int | tuple = None,
            use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Gives a new shape to an array without changing its data."""

    if ten.device == 'cpu':
        array = np.reshape(a=ten.data, newshape=new_shape)
    else:
        array = cp.reshape(a=ten.data, newshape=new_shape)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def rot90(ten: Tensor, k: int = None, axes=(0, 1),
          use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Rotate an array by 90 degrees in the plane specified by axes."""

    if ten.device == 'cpu':
        array = np.rot90(m=ten.data, k=k, axes=axes)
    else:
        array = cp.rot90(m=ten.data, k=k, axes=axes)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def block(tensors: Tensor,
          use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Assemble an nd-array from nested lists of blocks."""

    tensors_data = []
    for tensor_data in tensors:
        tensors_data.append(tensor_data.data)

    if tensors[0].device == 'cpu':
        array = np.block(arrays=tensors_data)
    else:
        array = cp.block(arrays=tensors_data)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def load(filename: str, use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Load a tensor from a file."""
    array = np.load(filename)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def save(filename: str, ten: Tensor, use_grad: bool = False,
         device: str = 'cpu') -> None:
    """Save a tensor to a file."""
    np.save(filename, ten.data)


def exp(ten: Tensor, use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Calculate the exponential of all elements in the input array."""
    if ten.device == 'cpu':
        array = np.exp(ten.data)
    else:
        array = cp.exp(ten.data)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def abs(ten: Tensor, use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Calculate the absolute value element-wise."""
    if ten.device == 'cpu':
        array = np.abs(ten.data)
    else:
        array = cp.abs(ten.data)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def log(ten: Tensor, use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Calculate the natural logarithm of all elements in the input array."""
    if ten.device == 'cpu':
        array = np.log(ten.data)
    else:
        array = cp.log(ten.data)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def sin(ten: Tensor, use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Calculate the sine of all elements in the input array."""
    if ten.device == 'cpu':
        array = np.sin(ten.data)
    else:
        array = cp.sin(ten.data)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def cos(ten: Tensor, use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Calculate the cosine of all elements in the input array."""
    if ten.device == 'cpu':
        array = np.cos(ten.data)
    else:
        array = cp.cos(ten.data)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def tan(ten: Tensor, use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Calculate the tangent of all elements in the input array."""
    if ten.device == 'cpu':
        array = np.tan(ten.data)
    else:
        array = cp.tan(ten.data)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def sinh(ten: Tensor, use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Calculate the hyperbolic sine of all elements in the input array."""
    if ten.device == 'cpu':
        array = np.sinh(ten.data)
    else:
        array = cp.sinh(ten.data)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def cosh(ten: Tensor, use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Calculate the hyperbolic cosine of all elements in the input array."""
    if ten.device == 'cpu':
        array = np.cosh(ten.data)
    else:
        array = cp.cosh(ten.data)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def tanh(ten: Tensor, use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Calculate the hyperbolic tangent of all elements in the input array."""
    if ten.device == 'cpu':
        array = np.tanh(ten.data)
    else:
        array = cp.tanh(ten.data)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def prod(ten: Tensor, axis: int = None, dtype: str = None,
         use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Return the product of array elements over a given axis."""
    if ten.device == 'cpu':
        array = np.prod(a=ten.data, axis=axis, dtype=dtype)
    else:
        array = cp.prod(a=ten.data, axis=axis, dtype=dtype)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def sum(ten: Tensor, axis: int = None, dtype: str = None,
        use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Sum of array elements over a given axis."""
    if ten.device == 'cpu':
        array = np.sum(a=ten.data, axis=axis, dtype=dtype)
    else:
        array = cp.sum(a=ten.data, axis=axis, dtype=dtype)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def cumprod(ten: Tensor, axis: int = None, dtype: str = None,
            use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Return the cumulative product of elements along a given axis."""
    if ten.device == 'cpu':
        array = np.cumprod(a=ten.data, axis=axis, dtype=dtype)
    else:
        array = cp.cumprod(a=ten.data, axis=axis, dtype=dtype)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def cumsum(ten: Tensor, axis: int = None, dtype: str = None,
           use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Return the cumulative sum of the elements along a given axis."""
    if ten.device == 'cpu':
        array = np.cumsum(a=ten.data, axis=axis, dtype=dtype)
    else:
        array = cp.cumsum(a=ten.data, axis=axis, dtype=dtype)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def max(ten: Tensor, axis: int = None, dtype: str = None,
        use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Return the maximum of an array or maximum along an axis."""
    if ten.device == 'cpu':
        array = np.max(a=ten.data, axis=axis)
    else:
        array = cp.max(a=ten.data, axis=axis, dtype=dtype)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def min(ten: Tensor, axis: int = None, dtype: str = None,
        use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Return the minimum of an array or minimum along an axis."""
    if ten.device == 'cpu':
        array = np.min(a=ten.data, axis=axis)
    else:
        array = cp.min(a=ten.data, axis=axis, dtype=dtype)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def convolve(ten1: Tensor, ten2: Tensor, mode: str = 'full',
             use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Returns the discrete, linear convolution of two one-dimensional sequences."""
    if ten1.device == 'cpu':
        array = np.convolve(a=ten1.data, v=ten2.data, mode=mode)
    else:
        array = cp.convolve(a=ten1.data, v=ten2.data, mode=mode)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)


def sign(ten: Tensor, use_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Returns an element-wise indication of the sign of a number."""
    if ten.device == 'cpu':
        array = np.sign(ten.data)
    else:
        array = cp.sign(ten.data)

    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad, device=device)





