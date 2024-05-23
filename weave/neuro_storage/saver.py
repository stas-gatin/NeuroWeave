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

import h5py
import os


def save_model(model, file_path=None, overwrite=False):
    """
        Save a neural network model to an HDF5 file.

        Parameters:
        model : model object, which must have 'weights' and 'config' attributes.
        file_path : string, the path to the file where the model will be saved.
        overwrite : bool, determines whether to overwrite the file if it already exists.
    """

    if not isinstance(model, dict) or 'weights' not in model or 'config' not in model:
        raise TypeError("Model must be a dictionary with 'weights' and 'config' keys.")

    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string.")

    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")

    if file_path is None or not file_path.strip():
        raise ValueError("File path is required to save the model. Set file_path='path/to/save'.")

    if not overwrite and os.path.exists(file_path):
        raise ValueError("File already exists. Set overwrite=True to overwrite it.")

    try:
        with h5py.File(file_path, 'w' if overwrite else 'x') as file:
            weight_group = file.create_group('weights')
            for i, weight in enumerate(model['weights']):
                weight_group.create_dataset(name=str(i), data=weight)

            config_group = file.create_group('config')
            for key, value in model['config'].items():
                config_group.attrs[key] = value

            file_size = os.path.getsize(file_path)

    except IOError as e:
        raise IOError(f"Failed to write to file '{file_path}': {str(e)}")
