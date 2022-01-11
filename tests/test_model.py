import os
import numpy as np
import torch
import sys
sys.path.insert(1, "src/models")
from model import MyAwesomeModel
import pytest
import unittest


class TestClassModel(unittest.TestCase):

    def testOne(self):
        model = MyAwesomeModel()
        model_name = "trained_model.pt"
        state_dict = torch.load("models/" + model_name)
        model.load_state_dict(state_dict)
        #Test if model output has the correct shape
        train_set = torch.load("data/processed/trainloader.pth")
        for images, labels in train_set:
            ps = model(images.unsqueeze(1))
            assert ps.size() == torch.Size([train_set.batch_size, len(np.unique(labels))]), "The output size does not match the expected"
            break

    #Try to further experiment with this one
   # @pytest.mark.parametrize("test_input,expected", [("3+5", 8), ("2+4", 6), ("6*9", 42)])
   # def test_eval(self, test_input, expected):
   #     assert eval(test_input) == expected

    def test_raise_value_error(self):
        #Check if error is raised with wrong input dimensions
        model = MyAwesomeModel()
        x = torch.rand(30,3,20,20)
        #self.assertRaises(ValueError, model(), torch.rand(30, 3, 20, 20))
        with self.assertRaises(ValueError):
            model(x)