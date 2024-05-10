# weave/__init__.py
import os
os.environ['CUDA_PATH'] = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/'
from .neuro_storage.saver import save_model
from .neuro_storage.loader import load_model
from weave.tensor import Tensor

__version__ = '1.0.0'
