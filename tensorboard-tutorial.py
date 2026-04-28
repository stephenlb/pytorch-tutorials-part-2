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
    transforms.Resize((200, 200)), ## TODO test if this will fix our loss reduction
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
                                              batch_size=8,
                                              shuffle=True)


validation_loader = torch.utils.data.DataLoader(validation_set,
                                                batch_size=8,
                                                shuffle=False)

# Class labels
#classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
#classes = ('cat', 'dog')



class Net(torch.nn.Module):
    def __init__(self, classes):
        super(Net, self).__init__()
        self.pool = torch.nn.MaxPool2d(4, 4)
        self.c1   = torch.nn.Conv2d(3,  6,  3)
        self.c2   = torch.nn.Conv2d(6,  20, 3)
        self.c3   = torch.nn.Conv2d(20, 20, 3)
        self.c4   = torch.nn.Conv2d(20, 20, 3)
        self.l1   = torch.nn.Linear(20 * 11 * 11, 120)
        self.l2   = torch.nn.Linear(120, 120)
        self.l3   = torch.nn.Linear(120, classes)

    def forward(self, inputs):
        ## TODO test this here to find a better combo
        #out = F.gelu(self.c1(inputs))
        out = self.pool(F.gelu(self.c1(inputs)))
        #out = F.gelu(self.c3(out))
        out = self.pool(F.gelu(self.c2(out)))
        #print(out.shape)
        #return out
        out = out.view(-1, 20 * 11 * 11) ## Flatten the 2d matrix to 1d vectors
        out = F.gelu(self.l1(out))
        out = F.gelu(self.l2(out))
        out = self.l3(out)
        return out

# Helper function for inline image display
# Extract a batch of 4 images
device = torch.accelerator.current_accelerator()
classes = 37
epochs = 20000
model = Net(classes).to(device)
writer = SummaryWriter('runs/exp1')
criterion = torch.nn.CrossEntropyLoss()
#criterion = torch.nn.NLLLoss()
learning_rate = 0.001
#optimizer = optim.Muon(model.parameters(), lr=learning_rate)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
losses = []

## TODO 
## TODO  finish the tensor board tutorial
## TODO 
## TODO  one hot output
## TODO 
## TODO  RELU activation - this wasn't as good as gelu
## TODO  kernel size of 3
## TODO  reduce images

## TODO  - these didn't work
## TODO  MUON - sad face this only works for 2d data and we are using 3d data
## TODO  NLLLoos working
## TODO 

## TODO DONE
## TODO tensor board output loss graph ✅
## TODO update model convolutions and ✅
## TODO fix model ✅
## TODO visualize the loss in TensorBoard - for tutorial 
## TODO batch incrase ✅
## TODO avg loss ✅
## TODO use GPU accelerator ✅
for epoch in range(epochs):
    print(f'epoch number {epoch} starting!')
    dataiter = iter(training_loader)
    for batch, (images, labels) in enumerate(dataiter):
        optimizer.zero_grad()
        out = model(images.to(device))
        loss = criterion(out, labels.to(device))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        #print(out)
        #print(out.shape)
        if batch % 50 == 0:
            #losses[1:] * target + (1-target) * losses[:-1]
            #a = 0.5
            #y[1:] * a + (1-a) y[:-1]
            loss_avg = sum(losses) / len(losses)
            print('loss',loss_avg, 'epoch', epoch)
            graph = {'Loss': loss_avg}
            writer.add_scalars('Model Training Loss', graph, epoch)
            writer.flush()
            break
        #if batch % 100 == 0:
        #    img_grid = torchvision.utils.make_grid(images)
        #    writer.add_image(f'{epoch}: Cat or Dog: {labels}' , img_grid)
writer.close()
