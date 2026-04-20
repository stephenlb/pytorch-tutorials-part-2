import torch
a = torch.rand(1, 5, 5) * 20 + 5
print(a)


norm = torch.nn.BatchNorm1d(5)
soft = torch.nn.Softmax(dim=1)
b = norm(a)
print(b)
c = soft(a)
print(c)
print('a.mean()',a.mean())  ## 15
print('b.mean()',b.mean())  ## == 0 1e-8
print('c.sum()',c.sum(dim=1))    ## == 1
print('c.mean()',c.mean(dim=1).sum())  ## == 0.5



#my_tensor = torch.rand(1, 4, 4) * 20 + 5
#print(my_tensor)

#print(my_tensor.mean())

#norm_layer = torch.nn.BatchNorm1d(4)
#normed_tensor = norm_layer(my_tensor)
#rint(normed_tensor)

#rint(normed_tensor.mean())

