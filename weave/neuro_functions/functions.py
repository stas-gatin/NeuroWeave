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
    'any'

]

# functions for creating tensors


# this function creates a tensor filled with ones
def ones(shape, dtype: str = None,
         use_grad: bool = False, device: str = 'cpu'):
    array = np.ones(shape, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


# this function creates a tensor filled with ones with the same shape as the input tensor
def ones_like(tensor, dtype: str = None, shape=None,
         use_grad: bool = False, device: str = 'cpu'):
    array = np.ones_like(a=tensor.data, dtype=dtype, shape=shape)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


# this function creates a tensor filled with empty values
def empty(shape, dtype: str = None,
          use_grad: bool = False, device: str = 'cpu'):
    array = np.empty(shape, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


# this function creates a tensor filled with a specific value
def full(shape, fill_value=np.inf, dtype=None,
         use_grad: bool = False, device: str = 'cpu'):
    array = np.full(shape, fill_value=fill_value, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


# this function creates a tensor filled with a specific value with the same shape as the input tensor
def full_like(tensor, fill_value=np.inf, dtype: str = None, shape=None,
         use_grad: bool = False, device: str = 'cpu'):
    array = np.full_like(a=tensor.data, fill_value=fill_value, dtype=dtype, shape=shape)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


# this function creates a tensor filled with zeros
def zeros(shape, dtype: str = None,
          use_grad: bool = False, device: str = 'cpu'):
    array = np.zeros(shape, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


# this function creates a tensor filled with zeros with the same shape as the input tensor
def zeros_like(tensor, dtype: str = None, shape=None,
         use_grad: bool = False, device: str = 'cpu'):
    array = np.zeros_like(a=tensor.data, dtype=dtype, shape=shape)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


# this function creates a tensor filled with random values
def rand(*shape,
         use_grad: bool = False, device: str = 'cpu'):
    array = np.random.rand(*shape)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


# this function creates a tensor  with ones on the diagonal and zeros elsewhere
def eye(rows, columns=None, k=0, dtype=None,
        use_grad: bool = False, device: str = 'cpu'):
    array = np.eye(rows, columns, k, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


# this function creates a tensor with evenly spaced values with careful handling of endpoints.
def linspace(start, stop, num, dtype=None,
    use_grad: bool = False, device: str = 'cpu'):
    array = np.linspace(start, stop,num,dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


# this function is similar to linspace, but uses a step size (instead of the number of samples).
def arange(start_, stop_, step, dtype=None,
           use_grad: bool = False, device: str = 'cpu'):
    array = np.arange(start_, stop_, step, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


# functions for tensor operations
# this function computes the dot product of two arrays
def dot(a,b, use_grad: bool = False, device: str = 'cpu'):
    array = np.dot(a,b)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


# this function computes the matrix product of two arrays
def matmul(a,b, use_grad: bool = False, device: str = 'cpu'):
    array = np.matmul(a,b)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


# this function computes the inner product of two arrays
def inner(a,b, use_grad: bool = False, device: str = 'cpu'):
    array = np.inner(a,b)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


# this function computes the outer product of two arrays
def outer(a,b, use_grad: bool = False, device: str = 'cpu'):
    array = np.outer(a,b)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


# this function computes the tensor dot product of two arrays
def tensordot(a,b, use_grad: bool = False, device: str = 'cpu'):
    array = np.tensordot(a,b)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


# this function computes the Einstein summation convention on the operands
def einsum(subscripts, *operands, use_grad: bool = False, device: str = 'cpu'):
    array = np.einsum(subscripts, *operands)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


# this function computes the sum along the diagonal
def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None, use_grad: bool = False, device: str = 'cpu'):
    array = np.trace(a, offset, axis1, axis2, dtype, out)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


# this function computes the norm of a tensor
def norm(x, ord=None, axis=None, keepdims=False, use_grad: bool = False, device: str = 'cpu'):
    array = np.linalg.norm(x, ord, axis, keepdims)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


# this function computes the inverse of a n x n tensor
def inv(a, use_grad: bool = False, device: str = 'cpu'):
    array = np.linalg.inv(a)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


# this function computes the inverse of a a multidimensional tensor
def tensorinv(a, ind=2, use_grad: bool = False, device: str = 'cpu'):
    array = np.linalg.tensorinv(a, ind)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


# this function computes the logical AND of a tensor
def all(x, axis=None, out=None, keepdims=False, use_grad: bool = False, device: str = 'cpu'):
    array = np.all(x, axis, out, keepdims)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


# this function computes the logical OR of a tensor
def any(x, axis=None, out=None, keepdims=False, use_grad: bool = False, device: str = 'cpu'):
    array = np.any(x, axis, out, keepdims)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)
















