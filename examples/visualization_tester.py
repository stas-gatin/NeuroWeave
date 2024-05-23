import weave
from weave import WeaveWeights

ten1 = weave.Tensor(data=[1, 2, 3])

visualizer = WeaveWeights()
visualizer.set_weights(ten1, ['Layer1', 'Output'])
visualizer.visualize()
