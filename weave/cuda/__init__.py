import warnings
try:
    import cupy as cp
    CUDA_IMPORT_FAILED = False
except (ImportError, ModuleNotFoundError):
    CUDA_IMPORT_FAILED = True
except UserWarning:
    warnings.filterwarnings("ignore")
    CUDA_IMPORT_FAILED = True
from .tools import Device, CUDADeviceCountError, CUDANotAvailableError


def is_available() -> bool:
    if CUDA_IMPORT_FAILED:
        return cp.cuda.is_available()
    return False


__all__ = ['is_available', 'CUDA_IMPORT_FAILED']