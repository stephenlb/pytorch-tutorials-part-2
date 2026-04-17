import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cjson import Config

dictionary = {}#Config(config_name='lstm-dictionary.json') ## Features
tags = {}#Config(config_name='lstm-tags.json') ## Labels

def build_tags(data):
    tags.update({"mask": 0})
    for sample in data:
        offset = len(tags)
        tags.update({
            tag: index + offset
            for index, tag in enumerate(set(sample[1]))
            if not(tag in tags)
        })

def build_dictionary(data):
    dictionary.update({"mask": 0})
    for sample in data:
        for index, word in enumerate(sample[0].lower().split()):
            if word in dictionary: continue
            dictionary.update({
                word: len(dictionary)
            })

def build_dictionary_character_level(data):
    dictionary.update({"mask": 0})
    for sample in data:
        for char in list(sample[0].lower()):
            if char in dictionary: continue
            dictionary.update({
                char: len(dictionary)
            })



def vectorize(data: list):
    max_sentence_length = max([len(x[0].split()) for x in data])
    print(f'{max_sentence_length=}')

    normalized = [
        [word < len(sentence[0].split()) and sentence[0].lower().split()[word] or ""
            for word in range(max_sentence_length)]
        for sentence in data
    ]

    max_word_length = max(max(len(x) for x in s) for s in normalized)
    print(f'{max_word_length=}')

    features = [

        [[c < len(word) and dictionary[word[c]] or dictionary['mask']
            for c in range(max_word_length)]

        for word in sentence]
        for sentence in normalized
    ]


    labels = [
        [tag < len(y[1]) and tags[y[1][tag]] or tags['mask']
            for tag in range(max_sentence_length)]
        for y in data]

    old_labels = [
        [[tag < len(y[1]) and (tags[y[1][tag]] == t and 1) or tags['mask'] for t in range(len(tags))]
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
        ## TODO CHAR LEVEL LSTM
        self.lstm_letter = nn.LSTM(embedding_dims, hidden_dims)
        self.lstm_word = nn.LSTM(hidden_dims, hidden_dims)
        self.l1 = nn.Linear(hidden_dims, hidden_dims)

    def forward(self, sentence):
        #this but the first dim is sentance
        out    = self.embedding(sentence)
        out, _ = self.lstm_letter(out)
        # print(f'shape letter output {out.shape}')
        out, _ = self.lstm_word(out.mean(1)[None])
        # print(f'shape word output {out.shape}')
        out    = self.l1(out[0])
        out    = torch.nn.functional.log_softmax(out, dim=0)
        return out

## Prepare Data
build_dictionary_character_level(training_data)
print("dictionary")
print(dictionary)

build_tags(training_data)
print("tags")
print(tags)

## Training Phase
EMBEDDING_DIMS = 12
HIDDEN_DIMS = 6
EPOCHS = 3000
learning_rate = 0.01
model = LSTMTagger(EMBEDDING_DIMS, HIDDEN_DIMS, len(dictionary), len(tags))
criterian = nn.NLLLoss()
optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

@torch.no_grad
def test_no_train():
    features, labels = vectorize(training_data)
    labels = torch.tensor(labels)
    features = torch.tensor(features)
    print(features[0])
    out = model(features[0])
    print(out)
    return
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
    features = torch.tensor(features)
    #print("features")
    #print(features)
    #print("labels")
    #print(labels)
    for epoch in range(EPOCHS):
        for index in range(len(labels)):
            target = labels[index]
            sentence = features[index]
            #print(target)
            #print(sentence)
            model.zero_grad()
            out = model(sentence)
            out = out.squeeze(1)
            #print("out")
            #print(out)
            print(out.shape)
            #print("target")
            #print(target)
            print(target.shape)
            delta = criterian(out, target)
            delta.backward()
            optim.step()
            print(delta.item())

#test_no_train()
train()