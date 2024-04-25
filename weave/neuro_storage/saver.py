import h5py
import os


def save_model(model, file_path, overwrite=False):
    """
    Save a neural network model to an HDF5 file.

    Parameters:
    model : model object, which must have 'weights' and 'config' attributes.
    file_path : string, the path to the file where the model will be saved.
    overwrite : bool, determines whether to overwrite the file if it already exists.
    """
    if not overwrite and os.path.exists(file_path):
        raise ValueError(
            "File already exists. Set overwrite=True to overwrite it.")

    with h5py.File(file_path, 'w') as file:
        # Creating a group for the weights in the file
        weight_group = file.create_group('weights')
        for i, weight in enumerate(model['weights']):
            weight_group.create_dataset(name=str(i), data=weight)

        # Creating a group for the configuration in the file
        config_group = file.create_group('config')
        for key, value in model['config'].items():
            config_group.attrs[key] = value
