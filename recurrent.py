import torch


class SPINDERMOON(torch.nn.Module):
    def __init__(self):
        super(SPINDERMOON, self).__init__()
        self.embedding_dim = 10
        self.hidden_dim = 100
        self.vocab_size = 100
        self.classes = 10
        self.dictionary = {}
        
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = torch.nn.LSTM(self.embedding_dim, self.hidden_dim)
        self.l1 = torch.nn.Linear(self.hidden_dim, self.classes)

    def lookup(self, sentence):
        words = sentence.split(' ')
        bag = {w:i for i,w  in enumerate(words)}
        self.dictionary.update(bag)
        vec = torch.tensor([[self.dictionary[w] for w in words]])
        return vec
        
    def forward(self, sentence):
        out = self.lookup(sentence)
        out = self.embedding(sentence)
        return out
        #l = 
        
model = SPINDERMOON()
sentence = 'hello Kyle and Jenlu'
vec = model.lookup(sentence)
print(sentence)
print(vec)

out = model(vec)
#print(out)
