# -*- coding: utf-8 -*-
import logging
import os
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from model import MyAwesomeModel
from torch.nn.functional import normalize
from torch.utils.data import TensorDataset

# parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# os.chdir(parent)
# grand_parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# os.chdir(grand_parent)

# @click.command()
# @click.argument('model_name', type=click.Path())
def train():
    # print("Training day and night")
    # logger = logging.getLogger(__name__)
    # logger.info('Training the model')

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    try:
        train_set = torch.load("data/processed/trainloader.pth")
    except:
        print("Remember to process the data before training the model")
        sys.exit(1)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.NLLLoss()
    epochs = 50
    running_loss = 0
    model.train()
    train_loss = []
    for e in range(epochs):
        for images, labels in train_set:
            # Adding dimension to get propor format tensor.shape() = (batch, channels, height, width)
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
