import torch

x = torch.nn.Linear(3,3) ## INSIDE YOUR MODEL
y = torch.rand(3,3)      ## for outside

print([r for r in x.parameters()])
print(y)

print(x.forward)
print(x(y))
