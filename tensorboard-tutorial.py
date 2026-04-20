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
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

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
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg)
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Extract a batch of 4 images
dataiter = iter(training_loader)

# Create a grid from the images and show them
#matplotlib_imshow(img_grid, one_channel=True)

writer = SummaryWriter('runs/exp1')

for i in range(20):
    images, labels = next(dataiter)
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image('Cat or Dog?', img_grid)
    images, labels = next(dataiter)
    writer.flush()


