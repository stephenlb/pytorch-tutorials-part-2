import torch
#def exp_add(x, y): return 2 * x.exp() + 3 * y 
#input = (torch.rand(3), torch.rand(3))
#print(input)
#print(torch.autograd.functional.jacobian(exp_add, input))


def double(x):
    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2
    return y

inputs = torch.rand(3)
gradients = torch.tensor([0.1, 1.0, 0.0001])
print(inputs)
print(torch.autograd.functional.vjp(double, inputs, v=gradients))

