# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

import torch
from torch.nn.functional import normalize
from torch.utils.data import TensorDataset
import numpy as np


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath: str, output_filepath: str):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data by normalizing')

    train = np.load(input_filepath + "/training_data.npz")
    images = normalize(torch.Tensor(train.f.images), dim=1)
    labels = torch.Tensor(train.f.labels).type(torch.LongTensor)
    trainset = TensorDataset(images, labels)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = np.load(input_filepath + "/test.npz")
    images_test = normalize(torch.Tensor(testset.f.images), dim=1)
    labels_test = torch.Tensor(testset.f.labels).type(torch.LongTensor)
    testset = TensorDataset(images_test, labels_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    torch.save(trainloader, output_filepath + "/trainloader.pth")
    torch.save(testloader, output_filepath + "/testloader.pth")



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
