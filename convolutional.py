import torch
import torch.nn.functional as F
from functools import reduce

class CNet(torch.nn.Module):
    def __init__(self):
        super(CNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1,  6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)

        self.l1 = torch.nn.Linear(16 * 6 * 6, 120)
        self.l2 = torch.nn.Linear(120, 84)
        self.l3 = torch.nn.Linear(84, 10)

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = F.max_pool2d(F.relu(out), (2,2))
        out = F.max_pool2d(F.relu(self.conv2(out)), 2)
        return out

        shape = out.size()[1:]
        height = reduce(lambda x, y: x * y, shape)
        out = out.view(-1, height)

        out = F.relu(self.l1(out))
        out = F.relu(self.l2(out))
        out = self.l3(out)

        return out

model = CNet()
inputs = torch.rand(1, 1, 32, 32)
out = model(inputs)
print(out.shape)

#shape = out.size()[1:]
#print(shape[0] * shape[1] * shape[2])
#print(reduce(lambda x, y: x * y, shape))
#print([l for l in shape])
#reduce
