import numpy as np
import torch


class TestClassData:
    def testOne(self):
        # Testing training data

        # Fetching training data
        train = np.load("data/raw/training_data.npz")
        images = torch.Tensor(train.f.images)
        labels = torch.Tensor(train.f.labels).type(torch.LongTensor)

        # Checking required proporties
        assert len(images) == 5000
        assert len(labels) == 5000
        assert images[0].size() == torch.Size([28, 28])
        assert np.array_equal(
            np.unique(labels), np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        )

    def testTwo(self):
        # Testing test data

        # Fetching test data
        testset = np.load("data/raw/test.npz")
        images_test = torch.Tensor(testset.f.images)
        test_labels = torch.Tensor(testset.f.labels).type(torch.LongTensor)

        assert len(images_test) == 5000
        assert len(test_labels) == 5000
        assert images_test[0].size() == torch.Size([28, 28])
        assert np.array_equal(
            np.unique(test_labels), np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        )
