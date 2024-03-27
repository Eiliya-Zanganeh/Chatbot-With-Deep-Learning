import os
import json
import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np


# nltk.download('punkt')


def tokenize(sentence):
    """
        "Hello World" -> ["Hello", "World"]
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
        "gaming" -> game
    """
    stemmer = PorterStemmer()
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
         tokenized_sentence -> How are you
         words -> ["i", "how", "am", "good", "hey", "are", "ok", "you"]
         return -> ["0", "1", "0", "0", "0", "1", "0", "1"]

    """
    tokenized_sentence = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, word in enumerate(words):
        if word in tokenized_sentence:
            bag[idx] = 1.0
    return bag


def load_sentence():
    folder_path = 'sentence'
    files = os.listdir(folder_path)

    data = {
        "intents": []
    }

    for file in files:
        with open(folder_path + '/' + file, "r") as f:
            key = "patterns"
            intent = {
                "tag": file.split(".")[0],
                "patterns": [],
                "responses": []
            }
            for line in f.readlines():
                line = line.strip()
                if line == "---":
                    key = "responses"
                    continue
                intent[key].append(line)
            data["intents"].append(intent)

    file_name = "data.json"

    with open(file_name, 'w') as file:
        json.dump(data, file)