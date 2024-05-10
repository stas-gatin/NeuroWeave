import numpy as np

from weave import neuro_functions

import weave

a = np.arange(9) - 4

b= weave.norm(a)

print(b)