import numpy as np
from weave import Tensor

__all__ = [
    'zeros',
]


def ones(shape, dtype: str = None):
    array = np.ones(shape, dtype=dtype)
    return Tensor(data=array, dtype=dtype)


def empty(shape, dtype: str = None):
    array = np.empty(shape, dtype=dtype)
    return Tensor(data=array, dtype=dtype)



def full(shape: int = None, fill_value=None, dtype: str = np.inf):
    array = np.full(shape, dtype=dtype)
    return Tensor(data=array, dtype=dtype)


def rand(shape, dtype=None, use_grad: bool = False, device:str = 'cpu'):
    array = np.random.rand(shape, dtype)
    return Tensor(data=array, dtype=dtype)


def zeros(shape, dtype: str = None, use_grad: bool = False, device:str = 'cpu'):
    array = np.zeros(shape, dtype=dtype)
    return Tensor(data=array, dtype=dtype)

