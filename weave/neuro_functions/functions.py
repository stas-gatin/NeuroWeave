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
    'arange'
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


