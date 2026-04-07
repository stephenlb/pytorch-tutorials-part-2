import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

lstm = nn.LSTM(3,3)
## Short term memory
hidden = (torch.rand(1, 1, 3), torch.randn(1, 1, 3))
inputs = [torch.rand(1,3) for _ in range(5)]
print(inputs)
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
print(inputs)

out, hidden = lstm(inputs, hidden)

##print(out)
##print(hidden)


#for i in inputs:
#    view = i.view(1, 1, -1)
#    out, hidden = lstm(view, hidden)
#    #print(f'{view=}')
#    print(f'{out=}')
#    print(f'{hidden=}')





