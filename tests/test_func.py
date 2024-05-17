import numpy as np

from weave import neuro_functions, Tensor

import weave
t = weave.tensor([1,2,3])
s = weave.tensor([0,1,0.5])
a = weave.convolve(t,s)
print(a)