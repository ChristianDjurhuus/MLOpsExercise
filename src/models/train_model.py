# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import torch
from model import MyAwesomeModel


def train():

    model = MyAwesomeModel()
    train_set = torch.load("data/processed/trainloader.pth")

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.NLLLoss()
    epochs = 50
    running_loss = 0
    model.train()
    train_loss = []
    for e in range(epochs):
        for images, labels in train_set:
            images = images.unsqueeze(1)
            optimizer.zero_grad()
            ps = model(images)
            loss = criterion(ps, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            # print statistics
            running_loss += loss.item()

        print("[%d] loss: %.3f" % (e + 1, running_loss / len(train_set)))
        running_loss = 0.0

    plt.plot(train_loss, label="Training loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("src/visualization/train_loss.png")

    torch.save(model.state_dict(), "models/trained_model.pt")


if __name__ == "__main__":
    train()
