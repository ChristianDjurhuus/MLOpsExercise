# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

import sys

import torch
import matplotlib.pyplot as plt
from torch.nn.functional import normalize
from torch.utils.data import TensorDataset
import numpy as np

from model import MyAwesomeModel


@click.command()
@click.argument('model_name', type=click.Path())
def train(model_name: str):
    print("Training day and night")
    logger = logging.getLogger(__name__)
    logger.info('Training the model')

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

    torch.save(model.state_dict(), "models/" + model_name)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    train()
