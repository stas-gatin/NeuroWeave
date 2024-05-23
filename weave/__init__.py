# weave/__init__.py
import weave.cuda
from .neuro_storage.saver import save_model
from .neuro_storage.loader import load_model
from .neuro_dataset.main import *
from .tensor import Tensor
from .neuro_functions.functions import *
from .nn import *
import numpy as np

int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64
uint8 = np.uint8
uint16 = np.int16
uint32 = np.int32
uint64 = np.int64

float16 = np.float16
float32 = np.float32
float64 = np.float64
_float_types = [
    float16,
    float32,
    float64,
]
_types = [
    *_float_types,
    int8, int16, int32, int64, uint8, uint16, uint32, uint64,
]

if weave.cuda.is_available():  # add only the available types based on the GPU availability status
    _types.extend(weave.cuda._types)
    _float_types.extend(weave.cuda._float_types)

__version__ = '1.0.0'
