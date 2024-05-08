import numpy as np
from weave import Tensor


def ones():
    pass


def zeros(shape, dtype=float, use_grad: bool = False, device:str = 'cpu'):
    array = np.zeros(shape, dtype=dtype)
    return Tensor(data=array, dtype=dtype)

