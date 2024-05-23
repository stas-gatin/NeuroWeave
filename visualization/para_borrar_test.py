
import weave
from weave_weights import WeaveWeights

ten1 = weave.Tensor([1, 2, 3])

weights = [ten1]
visualizer = WeaveWeights()
visualizer.set_weights(weights, ['Layer 1', 'Output'])
visualizer.visualize()