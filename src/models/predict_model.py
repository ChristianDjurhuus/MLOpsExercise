# -*- coding: utf-8 -*-
import logging
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


@click.command()
@click.argument("model_name", type=click.Path())
def evaluate(model_name):
    print("Evaluating until hitting the ceiling")

    model = MyAwesomeModel()
    state_dict = torch.load("models/" + model_name)
    model.load_state_dict(state_dict)
    try:
        test_set = torch.load("data/processed/testloader.pth")
    except:
        print("Remember to process the data before training the model")
        sys.exit(1)

    model.eval()
    accuracies = []
    with torch.no_grad():
        for images, labels in test_set:
            images = images.unsqueeze(1)
            # images = images.view(images.shape[0], -1)
            ps = model(images)
            # ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            # print(f'Accuracy: {accuracy.item() * 100}%')
            accuracies.append(accuracy)
    print("Estimate of accuracy: ", np.mean(accuracies))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    evaluate()
