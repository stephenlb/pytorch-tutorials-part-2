import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.tensorboard import SummaryWriter


# Gather datasets and prepare them for consumption
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((200, 200)),
    transforms.Normalize((0.5,), (0.5,)),
])

# Store separate training and validations splits in ./data
training_set = torchvision.datasets.OxfordIIITPet('./data',
    download=True,
    #train=True,
    transform=transform)
validation_set = torchvision.datasets.OxfordIIITPet('./data',
    download=True,
    #train=False,
    transform=transform)

training_loader = torch.utils.data.DataLoader(training_set,
                                              batch_size=1,
                                              shuffle=True)


validation_loader = torch.utils.data.DataLoader(validation_set,
                                                batch_size=1,
                                                shuffle=False)

# Class labels
#classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
#classes = ('cat', 'dog')



class Net(torch.nn.Module):
    def __init__(self, classes=10):
        super(Net, self).__init__()
        self.pool = torch.nn.MaxPool2d(4, 4)
        self.c1   = torch.nn.Conv2d(3, 6, 10)
        self.c2   = torch.nn.Conv2d(3, 20, 6)
        self.c3   = torch.nn.Conv2d(3, 20, 6)
        self.c4   = torch.nn.Conv2d(3, 20, 6)
        self.l1   = torch.nn.Linear(20 * 48 * 48, 120)
        self.l2   = torch.nn.Linear(120, 120)
        self.l3   = torch.nn.Linear(120, classes)

    def forward(self, inputs):
        out = F.gelu(self.c1(inputs))
        out = self.pool(F.gelu(self.c2(inputs)))
        out = F.gelu(self.c3(inputs))
        out = self.pool(F.gelu(self.c4(inputs)))
        out = out.view(-1, 20 * 48 * 48) ## Flatten the 2d matrix to 1d vectors
        out = F.gelu(self.l1(out))
        out = F.gelu(self.l2(out))
        out = self.l3(out)
        return inputs

# Helper function for inline image display
# Extract a batch of 4 images
dataiter = iter(training_loader)
labelIterator = iter(training_loader)
writer = SummaryWriter('runs/exp1')
classes = 36
model = Net(classes)

for i in range(1):
    image, label = next(dataiter)
    out = model(image)
    print(out)
    print(out.shape)
    img_grid = torchvision.utils.make_grid(image)
    writer.add_image(f'{i}: Cat or Dog: {label}' , img_grid)
writer.flush()
