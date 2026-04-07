import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

dictionary = {}
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
