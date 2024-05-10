# weave/__init__.py
import os
os.environ['CUDA_PATH'] = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/'
from .neuro_storage.saver import save_model
from .neuro_storage.loader import load_model
from .neuro_dataset.main import *
from weave.tensor import Tensor
from .neuro_functions.functions import *

__version__ = '1.0.0'
