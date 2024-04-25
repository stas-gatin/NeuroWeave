import h5py


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
