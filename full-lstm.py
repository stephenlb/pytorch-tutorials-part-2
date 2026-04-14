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
        offset = len(tags)
        tags.update({
            tag: index + offset
            for index, tag in enumerate(set(sample[1]))
            if not(tag in tags)
        })

print(len(dictionary))
def build_dictionary(data):
    for sample in data:
        for index, word in enumerate(sample[0].lower().split()):
            if word in dictionary: continue
            dictionary.update({
                word: len(dictionary)
            })

def vectorize(data: list):
    max_sentence_length = max([len(x[0].split()) for x in data])
    print(f'{max_sentence_length=}')
    features = [
        [word < len(sentence[0].split()) and dictionary[sentence[0].lower().split()[word]] or 0
            for word in range(max_sentence_length)]
        for sentence in data
    ]

    labels = [
        [tag < len(y[1]) and tags[y[1][tag]] or 0
            for tag in range(max_sentence_length)]
        for y in data]

    old_labels = [
        [[tag < len(y[1]) and tags[y[1][tag]] or 0 for t in range(len(tags))]
            for tag in range(max_sentence_length)]
        for y in data]

    return features, labels

## Data Preparation 
training_data = [
    ("The dog ate the apple", ["determiner", "noun", "verb", "determiner", "noun"]),
    ("The crow fly to water", ["determiner", "noun", "verb", "determiner", "noun"]),
    ("The crow can fly to the water", ["determiner", "noun", "verb", "verb", "verb", "determiner", "noun"]),
]

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
        out, _ = self.lstm(out)
        #out, _ = self.lstm(out.view(len(sentences), 1, -1))
        out = self.l1(out)
        #out = self.l1(out.view(len(sentences), 1, -1))
        #return out
        out = torch.nn.functional.log_softmax(out, dim=0)

        return out

## Prepare Data
build_dictionary(training_data)
print("dictionary")
print(dictionary)

build_tags(training_data)
#print("tags")
#print(tags)

## Training Phase
EMBEDDING_DIMS = 12
HIDDEN_DIMS = 6
EPOCHS = 1
learning_rate = 0.1
model = LSTMTagger(EMBEDDING_DIMS, HIDDEN_DIMS, len(dictionary), len(tags))
criterian = nn.NLLLoss()
optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

@torch.no_grad
def test_no_train():
    features, labels = vectorize(training_data)
    out = model(torch.tensor(features[0]))
    print(out)
    out = model(torch.tensor(features[1]))
    print(out)
    print(torch.tensor(features[0]))
    print(torch.tensor(features[2]))
    print(torch.tensor(features[1]))
    print(dictionary)
    print(len(dictionary))
    out = model(torch.tensor(features[2]))
    print(out)

def train():
    model.train()
    features, labels = vectorize(training_data)
    labels = torch.tensor(labels)
    featuers = torch.tensor(features)
    for epoch in range(EPOCHS):
        model.zero_grad()
        out = model(featuers)
        for index in range(len(labels)):
            print(out[index])
            print(labels[index])
            delta = criterian(out[index], labels[index])
            print(delta)
        #print(out[0].shape)
        #print(labels[0].shape)
        #print(f'Epoch = {epoch+1}')
        #optim.

#test_no_train()
train()
