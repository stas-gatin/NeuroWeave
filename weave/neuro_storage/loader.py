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


def load_model(file_path):
    """
    Load a model from an HDF5 file.

    Parameters:
    file_path : string, the path to the file from which the model is being loaded.

    Returns:
    A dictionary with 'weights' and 'config'.
    """
    # Initialize an empty model dictionary
    model = {'weights': [], 'config': {}}

    if file_path is None or not file_path.strip():
        raise ValueError("File path is required to save the model. Set file_path='path/to/save'.")

    try:
        # Attempt to open the specified HDF5 file
        with h5py.File(file_path, 'r') as file:

            # Load the weights from the file
            weights_group = file['weights']
            model['weights'] = [weights_group[name][()] for name in weights_group]

            # Load the configuration from the file
            config_group = file['config']
            model['config'] = {key: config_group.attrs[key] for key in config_group.attrs}

    except OSError as e:
        raise IOError(f"Cannot open or read the file: {file_path}. Error: {e}")

    except KeyError as e:
        raise KeyError(f"Problem with data format in the file: {file_path}. Missing key: {e}")

    # Return the loaded model dictionary
    return model
