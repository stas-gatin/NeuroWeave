import h5py
import logging

logging.basicConfig(level=logging.INFO)


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
        logging.error("No file path provided.")
        raise ValueError("File path is required to save the model. Set file_path='path/to/save'.")

    try:
        # Attempt to open the specified HDF5 file
        with h5py.File(file_path, 'r') as file:
            logging.info(f"Opening the file {file_path} for reading model data.")

            # Load the weights from the file
            weights_group = file['weights']
            model['weights'] = [weights_group[name][()] for name in weights_group]
            logging.info("Weights have been loaded successfully.")

            # Load the configuration from the file
            config_group = file['config']
            model['config'] = {key: config_group.attrs[key] for key in config_group.attrs}
            logging.info("Configuration has been loaded successfully.")

    except OSError as e:
        logging.error(f"Failed to read the file {file_path}: {e}")
        raise IOError(f"Cannot open or read the file: {file_path}. Error: {e}")

    except KeyError as e:
        logging.error(f"Key error in loading data from the file {file_path}: {e}")
        raise KeyError(f"Problem with data format in the file: {file_path}. Missing key: {e}")

    # Return the loaded model dictionary
    logging.info(f"Model loaded successfully from {file_path}")

    return model
