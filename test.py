import json
import random
import torch
from utils import tokenize, bag_of_words

data = torch.load("model.pth")

with open('data.json', 'r') as f:
    intents = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = data["model"].to(device)
words = data["words"]
tags = data["tags"]

model.eval()


def predict(sentence):
    sentence = tokenize(sentence)
    sentence = bag_of_words(sentence, words)
    sentence = sentence.reshape(1, sentence.shape[0])
    sentence = torch.from_numpy(sentence).to(device)
    output = model(sentence)
    _, predicted = torch.max(output.data, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(random.choice(intent["responses"]))
                break
    else:
        print("I don't understand :|")

    # print(tag)
    # print(tags)
    # print(prob)


while True:
    predict(input("Please enter a sentence: "))
