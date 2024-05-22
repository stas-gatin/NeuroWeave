from typing import Any
try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    pass


class CUDADeviceCountError(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class CUDANotAvailableError(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class Device:
    def __init__(self, device_type):
        self._device_type, *spec = device_type.split(':')
        if self._device_type == 'cuda' and spec:
            if not cp.cuda.is_available():
                raise CUDANotAvailableError('CUDA is not activated. Check if you posses a NVIDIA GPU and if so, '
                                            "investigate if it's CUDA compatible here: "
                                            "https://developer.nvidia.com/cuda-gpus. If it is, try installing the "
                                            "Nvidia CUDA Toolkit here: https://developer.nvidia.com/cuda-toolkit")
            if (v := int(*spec)) < 0 or v > cp.cuda.runtime.getDeviceCount() - 1:
                raise CUDADeviceCountError(f"Referred to a CUDA device that doesn't exist: CUDA:{v}")
            self._loc = cp.cuda.Device(v)
        elif self._device_type == 'cuda' and not spec:
            if not cp.cuda.is_available():
                raise CUDANotAvailableError('CUDA is not activated. Check if you posses a NVIDIA GPU and if so, '
                                            "investigate if it's CUDA compatible here: "
                                            "https://developer.nvidia.com/cuda-gpus. If it is, try installing the "
                                            "Nvidia CUDA Toolkit here: https://developer.nvidia.com/cuda-toolkit")
            self._loc = cp.cuda.Device(0)
        elif self._device_type == 'cpu':
            self._loc = 'CPU'
        else:
            raise AttributeError('Expected device type to be either "cpu" or "cuda".')
        self._device_type = self._device_type.upper()

    def switch(self, loc: int):
        if 0 <= loc < cp.cuda.runtime.getDeviceCount():
            self._loc = cp.cuda.Device(loc)
        else:
            raise CUDADeviceCountError(f"Referred to a CUDA device that doesn't exist: CUDA:{loc}")

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, (Device, str)), 'Cannot compare with classes other than Device or str.'
        if isinstance(other, Device):
            return str(self) == str(other)
        s, *num1 = str(self).split(':')
        o, *num2 = str(other).split(':')
        v1 = s.lower() == o.lower()
        try: num1.remove('0')
        except ValueError: pass
        try: num2.remove('0')
        except ValueError: pass
        v2 = True if (num1 and num2) or (not num1 and not num2) else False
        return v1 and v2

    def __str__(self):
        if isinstance(self._loc, str):
            return 'CPU'
        return f"CUDA:{self._loc.id}"

    def __repr__(self):
        if isinstance(self._loc, str):
            return f"weave.cuda.Device(CPU)"
        return f'weave.cuda.Device({self._device_type}:{self._loc.id})'


class NoCUDADevice:
    def __init__(self, device_type):
        self._device_type = device_type.split(':')[0]
        if self._device_type.lower() == 'cuda':
            raise CUDANotAvailableError("CUDA drivers haven't been found. Cannot put tensors in a inaccessible device.")
        elif self._device_type.lower() == 'cpu':
            self._loc = 'CPU'
        else:
            raise AttributeError('Expected device type to be either "cpu" or "cuda".')
        self._device_type = self._device_type.upper()

    def switch(self, loc: int):
        raise CUDANotAvailableError("CUDA drivers haven't been found. Cannot put tensors in a inaccessible device.")

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, (Device, str)), 'Cannot compare with classes other than Device or str.'
        if isinstance(other, Device):
            return str(self) == str(other)
        s, *num1 = str(self).split(':')
        o, *num2 = str(other).split(':')
        v1 = s.lower() == o.lower()
        try:
            num1.remove('0')
        except ValueError:
            pass
        try:
            num2.remove('0')
        except ValueError:
            pass
        v2 = True if (num1 and num2) or (not num1 and not num2) else False
        return v1 and v2

    def __str__(self):
        return "CPU"

    def __repr__(self):
        return 'weave.cuda.Device(CPU)'
