import unittest
import numpy as np
import torch
from sklearn import feature_selection

from minfnet.dats import input_generator as inge
from minfnet.dats import datasets as dase
from minfnet.ml import mime as modl
from minfnet.util import runtime_util as rtut

from heputl import logging as heplog

logger = heplog.get_logger(__name__)

class TestTrainOneThetaNoisyChannelPlainMinf(unittest.TestCase):

    def test_mi_increasing(self):
        # Sample data for testing
        noise_thetas = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        N_samples = 7000 # minimum number of samples needed for mime to give accurate estimate
        epochs = 200

        result_ll = np.zeros((len(noise_thetas), 3))

        for i,theta in enumerate(noise_thetas):

            A_train, B_train, *_ = inge.generate_noisy_channel_samples(N=N_samples, noise_std_nominal=theta, train_test_split=None)
            dataset_train = dase.MinfDataset(A_var=A_train, B_var=B_train)
            train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=1024, shuffle=True)

            # create model
            model = modl.MI_Model(B_N=1, acti='leaky', acti_out='tanh')
            model.to(rtut.device)

            # create optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            model.train()
                
            #****************************************#
            #               train model
            #****************************************#

            train_mi = modl.train(model, train_dataloader, epochs, optimizer)
            true_mi = feature_selection.mutual_info_regression(A_train.reshape(-1,1), B_train)[0]

            logger.info(f"Theta: {theta:.3f}, Train MI: {train_mi:.3f}, True MI: {true_mi:.3f}")

            result_ll[i] = theta, train_mi, true_mi

        train_mi = result_ll[:, 1]
        true_mi = result_ll[:, 2]

        for i in range(1, len(train_mi)):
            self.assertTrue((train_mi[i] > train_mi[i-1] and true_mi[i] > true_mi[i-1]) or
                            (train_mi[i] < train_mi[i-1] and true_mi[i] < true_mi[i-1]))

    def test_mi_within_squared_err(self):
        # Sample data for testing
        noise_thetas = np.array([1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1])
        N_samples = 10000 # minimum number of samples needed for mime to give accurate estimate
        epochs = 300
        squared_error_limit = 0.09

        result_ll = np.zeros((len(noise_thetas), 3))

        for i,theta in enumerate(noise_thetas):

            A_train, B_train, *_ = inge.generate_noisy_channel_samples(N=N_samples, noise_std_nominal=theta, train_test_split=None)
            dataset_train = dase.MinfDataset(A_var=A_train, B_var=B_train)
            train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=1024, shuffle=True)

            # create model
            model = modl.MI_Model(B_N=1, acti='leaky', acti_out='tanh')
            model.to(rtut.device)

            # create optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            model.train()
                
            #****************************************#
            #               train model
            #****************************************#

            train_mi = modl.train(model, train_dataloader, epochs, optimizer)
            true_mi = feature_selection.mutual_info_regression(A_train.reshape(-1,1), B_train)[0]

            logger.info(f"Theta: {theta:.3f}, Train MI: {train_mi:.3f}, True MI: {true_mi:.3f}")

            result_ll[i] = theta, train_mi, true_mi

        train_mi = result_ll[:, 1]
        true_mi = result_ll[:, 2]

        for i in range(len(train_mi)):
            squared_error = (train_mi[i] - true_mi[i]) ** 2
            self.assertLessEqual(squared_error, squared_error_limit)

if __name__ == '__main__':
    unittest.main()