import torch



a = torch.rand(1, 4, 4)
drop = torch.nn.Dropout(p=0.5)

print(a)
print(drop(a))
