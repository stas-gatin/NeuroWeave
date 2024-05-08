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
    'tensor',
    'diag',
    'tril',
    'triu',
    'concatenate',
]


def ones(shape, dtype: str = None,
         use_grad: bool = False, device: str = 'cpu'):
    array = np.ones(shape, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def ones_like(tensor_like, dtype: str = None, shape=None,
              use_grad: bool = False, device: str = 'cpu'):
    array = np.ones_like(a=tensor_like.data, dtype=dtype, shape=shape)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def empty(shape, dtype: str = None,
          use_grad: bool = False, device: str = 'cpu'):
    array = np.empty(shape, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def full(shape, fill_value=np.inf, dtype=None,
         use_grad: bool = False, device: str = 'cpu'):
    array = np.full(shape, fill_value=fill_value, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def full_like(tensor_like, fill_value=np.inf, dtype: str = None, shape=None,
              use_grad: bool = False, device: str = 'cpu'):
    array = np.full_like(a=tensor_like.data, fill_value=fill_value, dtype=dtype, shape=shape)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def zeros(shape, dtype: str = None,
          use_grad: bool = False, device: str = 'cpu'):
    array = np.zeros(shape, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def zeros_like(tensor_like, dtype: str = None, shape=None,
               use_grad: bool = False, device: str = 'cpu'):
    array = np.zeros_like(a=tensor_like.data, dtype=dtype, shape=shape)
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
    array = np.linspace(start, stop, num, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def arange(start_, stop_, step, dtype=None,
           use_grad: bool = False, device: str = 'cpu'):
    array = np.arange(start_, stop_, step, dtype=dtype)
    return Tensor(data=array, dtype=dtype, use_grad=use_grad)


def tensor(data, dtype=None, use_grad: bool = False, device: str = 'cpu'):
    return Tensor(data=data, dtype=dtype, use_grad=use_grad)


def diag(tensor_like, k=0, use_grad: bool = False, device: str = 'cpu'):
    array = np.diag(tensor_like.data, k)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def tril(tensor_like, k=0, use_grad: bool = False, device: str = 'cpu'):
    array = np.tril(tensor_like.data, k)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def triu(tensor_like, k=0, use_grad: bool = False, device: str = 'cpu'):
    array = np.triu(tensor_like.data, k)
    return Tensor(data=array, dtype=array.dtype, use_grad=use_grad)


def concatenate(tensors: tuple = None, axis=0, dtype=None, device: str = 'cpu'):
    pass
