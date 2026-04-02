import torch


r'[a-z\w\s\t.].?*{2,}'


## Pytorch does support multiple GPU accelerators INTEL, NVIDIA, AMD, APPLE
#device = torch.device('cpu')
def tensor(w, h): return torch.rand(w, h, requires_grad=True)
device = torch.accelerator.current_accelerator()
print(device)

x = tensor(20, 3000).to(device)
y = tensor(20, 3000).to(device)
z = tensor(20, 3000).to(device)

with torch.autograd.profiler.profile() as prf:
    for _ in range(1000):
        z = (z / x) * y

print(prf.key_averages())#.table(sort_by='self_cpu_time_totle'))



