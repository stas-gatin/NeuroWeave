import unittest
from weave import Tensor
import numpy as np
t1 = Tensor(data=[[1, 1, 1], [2, 2, 2]])
t2 = Tensor(data=[[1, 1, 1], [2, 2, 2]])

print(t1+t2)

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Матрица:")
print(matrix)


"""
class TestTensor(unittest.TestCase):
    t1 = Tensor(data=[[1, 1, 1], [2, 2, 2]])
    t2 = Tensor(data=[[1, 1, 1], [2, 2, 2]])
    def test_add(self):
        self.assertEqual(t1+t2, [[2, 2, 2], [4, 4, 4]])
        
        """