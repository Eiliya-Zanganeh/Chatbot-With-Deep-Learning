import json
import numpy as np
from torch.utils.data import Dataset
from utils import tokenize, stem, bag_of_words


class ModelDataset(Dataset):
    def __init__(self, url):
        with open(url) as file:
            data = json.load(file)

        self.words = []
        self.tags = []
        xy = []

        for intent in data["intents"]:
            tag = intent["tag"]
            self.tags.append(tag)
            for pattern in intent["patterns"]:
                word = tokenize(pattern)
                self.words.extend(word)
                xy.append((word, tag))

        ignore_words = ["?", "!", ".", ","]
        self.words = [stem(word) for word in self.words if word not in ignore_words]
        self.words = sorted(set(self.words))
        tags = sorted(set(self.tags))

        X_train = []
        Y_train = []

        for pattern_sentence, tag in xy:
            x = bag_of_words(pattern_sentence, self.words)
            X_train.append(x)

            y = tags.index(tag)
            Y_train.append(y)

        self.X_train = np.array(X_train, dtype=np.float32)
        self.Y_train = np.array(Y_train, dtype=np.float32)

    def __getitem__(self, index):
        return self.X_train[index], self.Y_train[index]

    def __len__(self):
        return len(self.X_train)
