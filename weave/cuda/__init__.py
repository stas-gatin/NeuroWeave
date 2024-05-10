from .tools import Device, CUDADeviceCountError, CUDANotAvailableError
import cupy as cp


def is_available() -> bool:
    return cp.cuda.is_available()


__all__ = ['is_available']
