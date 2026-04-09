import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cjson import Config

dictionary = Config(config_name='lstm-dictionary.json') ## Features
tags = Config(config_name='lstm-tags.json') ## Labels
print(dictionary)
print(tags)


def build_tags(data):
    for sample in data:
        offset = len(tags) + 1
        tags.update({
            tag: index + offset
            for index, tag in enumerate(set(sample[1]))
            if not(tag in tags)
        })

def build_dictionary(data):
    for sample in data:
        offset = len(dictionary)
        dictionary.update({
            word: index + offset
            for index, word in enumerate(sample[0].lower().split(' '))
            if not(word in dictionary)
        })

## TODO
def vectorize(data):
    #features = [dictionary[x[0]] for x in training_data]
    #labels = [tags[y[1] for y in training_data]
    pass

## Data Preparation 
training_data = [
    ("The dog ate the apple", ["determiner", "noun", "verb", "determiner", "noun"]),
    ("The crow can fly to the water", ["determiner", "noun", "verb", "verb", "verb", "determiner", "noun"]),
]
build_dictionary(training_data)
print("dictionary")
print(dictionary)

build_tags(training_data)
print("tags")
print(tags)


EMBEDDING_DIM = 6
HIDDEN_DIM = 6


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dims, hidden_dims, vocab_size, num_classes):
        super(LSTMTagger, self).__init__()
        self.embedding_dims = embedding_dims
        self.hidden_dims    = hidden_dims
        self.vocab_size     = vocab_size
        self.num_classes    = num_classes ## how many things are we classifying

        self.embedding = nn.Embedding(vocab_size, embedding_dims)
        self.lstm = nn.LSTM(embedding_dims, hidden_dims)
        self.l1 = nn.Linear(hidden_dims, num_classes)

    def forward(self, sentences):
        out = self.embedding(sentences)
        out, _ = self.lstm(out.view(len(sentences), 1, -1))
        out = self.l1(out.view(len(sentences), 1, -1))
        out = nn.log_softmax(out, dim=1)
        return out





