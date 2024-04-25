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
        for i, weight in enumerate(model.weights):
            weight_group.create_dataset(name=str(i), data=weight)

        # Creating a group for the configuration in the file
        config_group = file.create_group('config')
        for key, value in model.config.items():
            config_group.attrs[key] = value


def load_model(file_path):
    """
    Load a model from an HDF5 file.

    Parameters:
    file_path : string, the path to the file from which the model is being loaded.

    Returns:
    A dictionary with 'weights' and 'config'.
    """
    model = {'weights': [], 'config': {}}
    with h5py.File(file_path, 'r') as file:
        weights_group = file['weights']
        model['weights'] = [weights_group[name][()] for name in weights_group]

        config_group = file['config']
        model['config'] = {key: config_group.attrs[key] for key in
                           config_group.attrs}

    return model
