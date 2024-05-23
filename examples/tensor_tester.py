from weave import Tensor

# Example of creating a Tensor
ten1 = Tensor(data=[1, 2, 3])
print(ten1.__repr__())
print(ten1.grad)
