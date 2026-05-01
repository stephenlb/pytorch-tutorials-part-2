import random
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


transform = transforms.Compose([
    transforms.ToTensor(),## Functional because it returns a FUNCITON
    transforms.Normalize((0.5,),(0.5,)),
])

training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

traning_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)

# Class labels
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

total = len(training_set) + len(validation_set)
print(f'Training set size: {len(training_set)} {round((len(training_set)/total) * 100.)}%')
print(f'Validation set size: {len(validation_set)} {round((len(validation_set)/total) * 100.)}%')


import matplotlib.pyplot as plt
import numpy as np

def imgshow(img):
    img = img.mean(dim=0)
    img = img / 2. + 0.5
    npimg = img.numpy()
    plt.imshow(npimg)
    plt.show()

#img_grid = torchvision.utils.make_grid(images)
#imgshow(images[0].squeeze(0))
#imgshow(images[0][0])
#imgshow(images[0,0,:,:])
#imgshow(images[0,0])
#imgshow(img_grid)

class GarmetClassifier(torch.nn.Module):
    def __init__(self):
        super(GarmetClassifier, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.lin1 = torch.nn.Linear(16 * 4 * 4, 120)
        self.lin2 = torch.nn.Linear(120, 84)
        self.lin3 = torch.nn.Linear(84, 10)
        self.gelu = torch.nn.GELU()
    
    def forward(self, inputs):
        out = self.pool(self.gelu(self.conv1(inputs)))
        out = self.pool(self.gelu(self.conv2(out)))
        out = self.gelu(self.lin1(out.view(-1, 16 * 4 * 4)))
        out = self.gelu(self.lin2(out))
        out = self.lin3(out)
        return out

epochs = 5
device = torch.accelerator.current_accelerator()
model = GarmetClassifier()
model = model.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=0.001)
optimizers = [
    torch.optim.SGD(model.parameters(), lr=0.001),
    torch.optim.Adam(model.parameters(), lr=0.001),
    torch.optim.AdamW(model.parameters(), lr=0.001),
    torch.optim.RMSprop(model.parameters(), lr=0.001),
]

def test():
    dataiter = iter(validation_loader)
    accurate = torch.tensor([])
    for batch, (images, labels) in enumerate(dataiter):
        out = model(images.to(device))
        answers = torch.argmax(out, dim=1)
        #print(answers)
        #print(labels.detach().cpu().numpy())
        accuracy = answers.detach().cpu() == labels
        accurate = torch.concat((accurate, accuracy))

    total_accuracy = (float(sum(accurate)) / float(len(accurate))) * 100.
    print(f'Accuracy: {total_accuracy:.2f}% ({len(accurate)})')

test()

### TODO multple Optimizers altogether ✅
### TODO finish the training with EPOCH ✅
### TODO finish the tutorial ( which includes tensorborad )
### TODO 

losses = []
def train():
    model.train()
    for epoch in range(epochs):
        dataiter = iter(traning_loader)
        for batch, (features, labels) in enumerate(dataiter):
            model.zero_grad()
            out = model(features.to(device))
            loss = loss_fn(out, labels.to(device))
            loss.backward()
            #optim.step()
            optimizers[random.randint(0,len(optimizers)-1)].step()
            losses.append(loss.item())

            if batch % 100 == 0:
                print(f'Epoch {epoch+1}', sum(losses)/len(losses))
            if batch % 1000 == 0:
                test()
                
        print(f'Finished Epoch {epoch+1}')

train()
#try:    train()
#except: pass


