from weave import Tensor
import weave.cuda
import matplotlib.pyplot as plt
import numpy as np
if weave.cuda.is_available():
    import cupy as cp


class WeaveWeights:
    def __init__(self):
        self.weights = []
        self.layer_names = []

    def set_weights(self, weights, layer_names=None):
        """
        Sets the weights of the neural network.

        Parameters:
        weights (list of weave.Tensor): List of weight tensors for each layer.
        layer_names (list of str): List of layer names.
        """
        # Check and extract the weights from the tensor
        if all(isinstance(weight, Tensor) for weight in weights):
            self.weights = [weight.data for weight in weights]
        else:
            raise ValueError('Weights must be part of the tensor')

        if layer_names is None:
            self.layer_names = [f'Layer {i+1}' for i in range(len(weights))]
        else:
            self.layer_names = layer_names

    def visualize(self):
        """
        Visualizes the weights of the neural network
        """

        num_layers = len(self.weights)

        # Create a subplot for each layer
        fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 5))

        if not isinstance(axes, list):
            axes = [axes]

        for i, (weight, layer_name, ax) in enumerate(zip(self.weights, self.layer_names, axes)):
            # Convert tensor to a numpy array for visualization
            if not isinstance(weight, np.ndarray):  # We consider the possibility if weights being on the CPU or the GPU.
                weight_np = cp.asnumpy(weight)
            else:
                weight_np = np.asarray(weight)

            im = ax.matshow(weight_np, cmap='viridis')

            ax.set_title('Weave Weights')
            ax.set_xlabel('Output')
            ax.set_ylabel('Input')

            fig.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.show()
