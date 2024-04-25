import weave
import numpy as np


# Test model setup
test_model = {
    'weights': [np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([0.5, 0.6])],  # Example weight arrays
    'config': {
        'learning_rate': 0.01,
        'activation': 'relu'
    }
}

weave.save_model(test_model, 'my_model.h5', overwrite=True)
