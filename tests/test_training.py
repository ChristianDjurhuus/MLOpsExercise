import sys
import unittest
import torch
sys.path.insert(1, "src/models")
from model import MyAwesomeModel

# https://thenerdstation.medium.com/how-to-unit-test-machine-learning-code-57cf6fd81765


class testClassTraining(unittest.TestCase):
    def testing_optimization(self):
        # Testing if model parameters actually changes
        model = MyAwesomeModel()
        orig_parameters = list(model.parameters())
        train_set = torch.load("data/processed/trainloader.pth")
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.NLLLoss()
        for images, labels in train_set:
            optimizer.zero_grad()
            ps = model(images.unsqueeze(1))
            loss = criterion(ps, labels)
            loss.backward()
            optimizer.step
            self.assertListEqual(
                orig_parameters, list(model.parameters())
            ), "The model parameters are not being optimized"
            break

    def test_loss(self):
        model = MyAwesomeModel()
        train_set = torch.load("data/processed/trainloader.pth")
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.NLLLoss()
        for images, labels in train_set:
            optimizer.zero_grad()
            ps = model(images.unsqueeze(1))
            loss = criterion(ps, labels)
            loss.backward()
            optimizer.step

            assert loss != 0, "The loss is stuck at zero"
