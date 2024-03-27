import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Model
from dataset import ModelDataset
from utils import load_sentence


def train():
    load_sentence()
    dataset = ModelDataset("data.json")

    batch_size = 8
    hidden_size = 8
    output_size = len(dataset.tags)
    input_size = len(dataset.X_train[0])
    lr = 0.001
    epochs = 1000

    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(input_size, hidden_size, output_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        for datas, labels in train_loader:
            datas = datas.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(datas)
            loss = criterion(outputs, labels.long())

            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"epoch: {epoch + 1}, loss: {loss.item():.4f}")

    # torch.save(model.state_dict(), "model.pth")
    data = {
        "model": model,
        "words": dataset.words,
        "tags": dataset.tags
    }
    torch.save(data, "model.pth")


if __name__ == "__main__":
    train()
