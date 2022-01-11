import os
import numpy as np
import torch

#Temporary fix
_PATH_DATA = os.path.join(
    "/Users/christiandjurhuus/Documents/DTU/6_semester/ml_ops/dtu_mlops/s1_getting_started/exercise_files/final_exercise/mlOperation_day2/",
    "data")  # root of data


class TestClassData:
    def testOne(self):
        #Testing training data

        #Fetching training data
        train = np.load(_PATH_DATA + "/raw/training_data.npz")
        images = torch.Tensor(train.f.images)
        labels = torch.Tensor(train.f.labels).type(torch.LongTensor)

        #Checking required proporties
        assert len(images) == 5000
        assert len(labels) == 5000
        assert images[0].size() == torch.Size([28, 28])
        assert np.array_equal(np.unique(labels), np.array([0,1,2,3,4,5,6,7,8,9]))




    def testTwo(self):
        #Testing test data

        #Fetching test data
        testset = np.load(_PATH_DATA + "/raw/test.npz")
        images_test = torch.Tensor(testset.f.images)
        test_labels = torch.Tensor(testset.f.labels).type(torch.LongTensor)

        assert len(images_test) == 5000
        assert len(test_labels) == 5000
        assert images_test[0].size() == torch.Size([28, 28])
        assert np.array_equal(np.unique(test_labels), np.array([0,1,2,3,4,5,6,7,8,9]))
