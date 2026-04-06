import torch
import torch.nn.functional as F


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

    def tokenize(self, sentences):
        vecs = []
        for sentence in sentences:
            words = sentence.split(' ')
            bag = {w:i for i,w  in enumerate(words)}
            self.dictionary.update(bag)
            vec = [self.dictionary[w] for w in words]
            vecs.append(vec)
        return torch.tensor(vecs)
        
    def forward(self, sentence):
        out = self.tokenize(sentence)
        wordcount = len(out[0])
        out = self.embedding(out)
        out, _ = self.lstm(out.view(wordcount, 1, -1))
        out = self.l1(out.view(wordcount, -1))
        out = F.log_softmax(out, dim=1)
        return out

model = SPINDERMOON()
sentence = 'hello Kyle and Jenlu'
vecs = model.tokenize([sentence])
print(sentence)
print(vecs)
out = model([sentence])
print(out)
print(out.shape)
