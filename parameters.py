import torch

class Tiny(torch.nn.Module):
    def __init__(self):
        super(Tiny, self).__init__()

        self.l1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(200, 10)
        self.outactivation = torch.nn.Softmax()
        self.seq = torch.nn.Sequential(
            self.l1,
            self.activation,
            self.l2,
            self.outactivation,
        )

    def forward(self, input):
        return self.seq(input)


model = Tiny()
#inputs = torch.rand(1, 100)
#out = model(inputs)
#print(out)

#print(model)
#print(model.l1)
#print(model.l1.parameters())
for p in model.parameters():
    print(p)
