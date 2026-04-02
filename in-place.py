import torch
import math

a = torch.linspace(0., 2., steps=25, requires_grad=True)
print(a)
torch.sin_(a)
print(a)

