import h5py
import os
import logging

logging.basicConfig(level=logging.INFO)


def save_model(model, file_path=None, overwrite=False):
    """
        Save a neural network model to an HDF5 file.

        Parameters:
        model : model object, which must have 'weights' and 'config' attributes.
        file_path : string, the path to the file where the model will be saved.
        overwrite : bool, determines whether to overwrite the file if it already exists.
    """

    if not isinstance(model, dict) or 'weights' not in model or 'config' not in model:
        logging.error("Invalid model format.")
        raise TypeError("Model must be a dictionary with 'weights' and 'config' keys.")

    if not isinstance(file_path, str):
        logging.error("Invalid file path type.")
        raise TypeError("file_path must be a string.")

    if not isinstance(overwrite, bool):
        logging.error("Invalid overwrite flag type.")
        raise TypeError("overwrite must be a boolean.")

    if file_path is None or not file_path.strip():
        logging.error("No file path provided.")
        raise ValueError("File path is required to save the model. Set file_path='path/to/save'.")

    if not overwrite and os.path.exists(file_path):
        logging.error("File exists but overwrite is set to False.")
        raise ValueError("File already exists. Set overwrite=True to overwrite it.")

    logging.info(f"Attempting to save model to {file_path}")
    try:
        with h5py.File(file_path, 'w' if overwrite else 'x') as file:
            weight_group = file.create_group('weights')
            for i, weight in enumerate(model['weights']):
                weight_group.create_dataset(name=str(i), data=weight)
            logging.info("Weights saved successfully.")

            config_group = file.create_group('config')
            for key, value in model['config'].items():
                config_group.attrs[key] = value
            logging.info("Configuration saved successfully.")

            file_size = os.path.getsize(file_path)
            logging.info(f"Model saved successfully at {file_path}, file size: {file_size} bytes")

    except IOError as e:
        logging.error(f"Failed to write to file '{file_path}': {str(e)}")
        raise IOError(f"Failed to write to file '{file_path}': {str(e)}")
