import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cjson import Config

dictionary = Config(config_name='lstm-dictionary.json')
print(dictionary)

tags = { w:i for i, w in enumerate(["determiner", "noun", "verb"])}
print("tags")
print(tags)

training_data = [
    ("The dog ate the apple", ["determiner", "noun", "verb", "determiner", "noun"]),
    ("The crow can fly to the water", ["determiner", "noun", "verb", "verb", "verb", "determiner", "noun"]),
]

def build_dictionary(data):
    for sample in data:
        sentence = sample[0]
        offest = len(dictionary)
        update = {
            word: offest + index for index,word in
            enumerate(sentence.lower().split(' '))
        }
        dictionary.update(update)

build_dictionary(training_data)
print("dictionary")
print(dictionary)


EMBEDDING_DIM = 6
HIDDEN_DIM = 6



class LSTMTagger(nn.Module):
    def __init__(self, embedding_dims, hidden_dims, vocab_size, num_classes):
        super(LSTMTagger, self).__init__()




