import warnings
try:
    import cupy as cp
    CUDA_IMPORT_FAILED = False
except (ImportError, ModuleNotFoundError):
    CUDA_IMPORT_FAILED = True
    from .tools import NoCUDADevice as Device
    from .tools import CUDADeviceCountError, CUDANotAvailableError
else:
    from .tools import Device, CUDANotAvailableError, CUDADeviceCountError

    int8 = cp.int8
    int16 = cp.int16
    int32 = cp.int32
    int64 = cp.int64
    uint8 = cp.uint8
    uint16 = cp.uint16
    uint32 = cp.uint32

    float16 = cp.float16
    float32 = cp.float32
    float64 = cp.float64
    _float_types = [
        float16,
        float32,
        float64,
    ]
    _types = [
        *_float_types,
        int8, int16, int32, int64, uint8, uint16, uint32,
    ]


def is_available() -> bool:
    if not CUDA_IMPORT_FAILED:
        try:
            return cp.cuda.is_available()
        except cp.cuda.runtime.CUDARuntimeError:
            return False
    return False


__all__ = ['is_available', 'CUDA_IMPORT_FAILED']

if __name__ != "__main__" and CUDA_IMPORT_FAILED:
    warnings.warn("The required CUDA drivers and folders haven't been found. CUDA availability for this library is"
                  " unavailable.")
