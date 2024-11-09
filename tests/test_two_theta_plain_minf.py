import unittest
import numpy as np
import torch
from sklearn import feature_selection
from minfnet.dats import input_generator as inge
from minfnet.dats import datasets as dase
from minfnet.ml import mime as modl
from minfnet.util import runtime_util as rtut
from minfnet.util import data_util as datu

class TestTrainTwoThetaNoisyChannelPlainMinf(unittest.TestCase):

    def test_mi_decreasing(self):

        N_theta = 6
        tt1, tt2 = datu.make_two_theta_grid(theta_min=0.2, theta_max=1.2, N=N_theta)
        N_samples = 7000 # minimum number of samples needed for mime to give accurate estimate
        epochs = 200

        result_arr = np.zeros((len(tt1.flatten()), 4))

        for i, (t1, t2) in enumerate(zip(tt1.flatten(), tt2.flatten())):

            A_train, B_train, *_ = inge.generate_two_theta_noisy_samples(N=N_samples, t1_noise_nominal=t1, t2_damp_nominal=t2)
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

            result_arr[i] = t1, t2, train_mi, true_mi


        # Sort by tt1
        tt1_sort, tt2_sort, train_mi_sort, true_mi_sort, _ = datu.sort_results_by_theta(result_arr, sort_by_idx=0)
        # Slice arrays into chunks of N_theta
        tt1_chunks, tt2_chunks, train_mi_chunks, true_mi_chunks = [np.array_split(aa, N_theta) for aa in [tt1_sort, tt2_sort, train_mi_sort, true_mi_sort]]

        for tt1_chunk, tt2_chunk, train_mi_chunk, true_mi_chunk in zip(tt1_chunks, tt2_chunks, train_mi_chunks, true_mi_chunks):
            
            sorted_indices_chunk = np.argsort(tt2_chunk)
            train_mi_chunk_sorted = train_mi_chunk[sorted_indices_chunk]
            true_mi_chunk_sorted = true_mi_chunk[sorted_indices_chunk]

            for i in range(1, len(train_mi_chunk_sorted)):
                self.assertTrue(train_mi_chunk_sorted[i] <= train_mi_chunk_sorted[i-1], 
                    f"train_mi_chunk_sorted is not decreasing at index {i}: {train_mi_chunk_sorted[i]} > {train_mi_chunk_sorted[i-1]}")
            for i in range(1, len(true_mi_chunk_sorted)):
                self.assertTrue(true_mi_chunk_sorted[i] <= true_mi_chunk_sorted[i-1], 
                    f"true_mi_chunk_sorted is not decreasing at index {i}: {true_mi_chunk_sorted[i]} > {true_mi_chunk_sorted[i-1]}")

        # Sort by tt2
        tt1_sort, tt2_sort, train_mi_sort, true_mi_sort, _ = datu.sort_results_by_theta(result_arr, sort_by_idx=1)
        # Slice arrays into chunks of N_theta
        tt1_chunks, tt2_chunks, train_mi_chunks, true_mi_chunks = [np.array_split(aa, N_theta) for aa in [tt1_sort, tt2_sort, train_mi_sort, true_mi_sort]]


        for tt2_chunk, tt1_chunk, train_mi_chunk, true_mi_chunk in zip(tt2_chunks, tt1_chunks, train_mi_chunks, true_mi_chunks):
            
            sorted_indices_chunk = np.argsort(tt1_chunk)
            train_mi_chunk_sorted = train_mi_chunk[sorted_indices_chunk]
            true_mi_chunk_sorted = true_mi_chunk[sorted_indices_chunk]

            for i in range(1, len(train_mi_chunk_sorted)):
                self.assertTrue(train_mi_chunk_sorted[i] <= train_mi_chunk_sorted[i-1], 
                    f"train_mi_chunk_sorted is not decreasing at index {i}: {train_mi_chunk_sorted[i]} > {train_mi_chunk_sorted[i-1]}")
            for i in range(1, len(true_mi_chunk_sorted)):
                self.assertTrue(true_mi_chunk_sorted[i] <= true_mi_chunk_sorted[i-1], 
                    f"true_mi_chunk_sorted is not decreasing at index {i}: {true_mi_chunk_sorted[i]} > {true_mi_chunk_sorted[i-1]}")


    def test_mi_within_squared_err(self):
        
        N_theta = 6
        tt1, tt2 = datu.make_two_theta_grid(theta_min=0.2, theta_max=1.2, N=N_theta)
        N_samples = 7000
        epochs = 200

        result_ll = np.zeros((len(tt1.flatten()), 4))

        for i, (t1, t2) in enumerate(zip(tt1.flatten(), tt2.flatten())):

            A_train, B_train, *_ = inge.generate_two_theta_noisy_samples(N=N_samples, t1_noise_nominal=t1, t2_damp_nominal=t2)
            dataset_train = dase.MinfDataset(A_var=A_train, B_var=B_train)
            train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=1024, shuffle=True)

            model = modl.MI_Model(B_N=1, acti='leaky', acti_out='tanh')
            model.to(rtut.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            model.train()

            train_mi = modl.train(model, train_dataloader, epochs, optimizer)
            true_mi = feature_selection.mutual_info_regression(A_train.reshape(-1,1), B_train)[0]

            result_ll[i] = t1, t2, train_mi, true_mi

        tt1 = result_ll[:, 0]
        tt2 = result_ll[:, 1]
        train_mi = result_ll[:, 2]
        true_mi = result_ll[:, 3]

        squared_errors = (train_mi - true_mi) ** 2
        mean_squared_error = np.mean(squared_errors)

        self.assertTrue(mean_squared_error < 0.07, f"Mean squared error is too high: {mean_squared_error}")

if __name__ == '__main__':
    unittest.main()