import torch

a = torch.rand(1, 6,6)
print(a)

pool = torch.nn.MaxPool2d((3,3), stride=1)
b = pool(a)
print(b)
