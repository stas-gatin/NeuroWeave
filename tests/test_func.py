import numpy as np

from weave import neuro_functions, Tensor

import weave
t = Tensor(data=[[1,2,3],[4,5,6]])
a = weave.min(t)
print(a)