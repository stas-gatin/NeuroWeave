from weave import Tensor
import random
import numpy as np


ten1 = Tensor(data=[1, 2, 3])
print(ten1.__repr__())

print(ten1.grad)

