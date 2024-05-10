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


def ones(shape, dtype: str = None,
         use_grad: bool = False, device: str = 'cpu'):
    array = np.ones(shape, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def ones_like(tensor, dtype: str = None, shape=None,
         use_grad: bool = False, device: str = 'cpu'):
    array = np.ones_like(a=tensor.data, dtype=dtype, shape=shape)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def empty(shape, dtype: str = None,
          use_grad: bool = False, device: str = 'cpu'):
    array = np.empty(shape, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def full(shape, fill_value=np.inf, dtype=None,
         use_grad: bool = False, device: str = 'cpu'):
    array = np.full(shape, fill_value=fill_value, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def full_like(tensor, fill_value=np.inf, dtype: str = None, shape=None,
         use_grad: bool = False, device: str = 'cpu'):
    array = np.full_like(a=tensor.data, fill_value=fill_value, dtype=dtype, shape=shape)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def zeros(shape, dtype: str = None,
          use_grad: bool = False, device: str = 'cpu'):
    array = np.zeros(shape, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def zeros_like(tensor, dtype: str = None, shape=None,
         use_grad: bool = False, device: str = 'cpu'):
    array = np.zeros_like(a=tensor.data, dtype=dtype, shape=shape)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)




def rand(*shape,
         use_grad: bool = False, device: str = 'cpu'):
    array = np.random.rand(*shape)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def eye(rows, columns=None, k=0, dtype=None,
        use_grad: bool = False, device: str = 'cpu'):
    array = np.eye(rows, columns, k, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def linspace(start, stop, num, dtype=None,
    use_grad: bool = False, device: str = 'cpu'):
    array = np.linspace(start, stop,num,dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def arange(start_, stop_, step, dtype=None,
           use_grad: bool = False, device: str = 'cpu'):
    array = np.arange(start_, stop_, step, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def dot(a,b, use_grad: bool = False, device: str = 'cpu'):
    array = np.dot(a,b)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def matmul(a,b, use_grad: bool = False, device: str = 'cpu'):
    array = np.matmul(a,b)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def inner(a,b, use_grad: bool = False, device: str = 'cpu'):
    array = np.inner(a,b)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def outer(a,b, use_grad: bool = False, device: str = 'cpu'):
    array = np.outer(a,b)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def tensordot(a,b, use_grad: bool = False, device: str = 'cpu'):
    array = np.tensordot(a,b)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def einsum(subscripts, *operands, use_grad: bool = False, device: str = 'cpu'):
    array = np.einsum(subscripts, *operands)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None, use_grad: bool = False, device: str = 'cpu'):
    array = np.trace(a, offset, axis1, axis2, dtype, out)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def norm(x, ord=None, axis=None, keepdims=False, use_grad: bool = False, device: str = 'cpu'):
    array = np.linalg.norm(x, ord, axis, keepdims)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def inv(a, use_grad: bool = False, device: str = 'cpu'):
    array = np.linalg.inv(a)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def tensorinv(a, ind=2, use_grad: bool = False, device: str = 'cpu'):
    array = np.linalg.tensorinv(a, ind)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def all(x, axis=None, out=None, keepdims=False, use_grad: bool = False, device: str = 'cpu'):
    array = np.all(x, axis, out, keepdims)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def any(x, axis=None, out=None, keepdims=False, use_grad: bool = False, device: str = 'cpu'):
    array = np.any(x, axis, out, keepdims)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)
















