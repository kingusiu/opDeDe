import numpy as np
import torch
from sklearn import feature_selection
import wandb

from minfnet.dats import input_generator as inge
from minfnet.dats import datasets as dase
from minfnet.ml import model as modl
from minfnet.util import runtime_util as rtut

from heputl import logging as heplog
import random
import matplotlib.pyplot as plt


logger = heplog.get_logger(__name__)

def plot_results(result_ll):

    result_ll = np.array(result_ll)
    result_ll = result_ll[result_ll[:, 0].argsort()]
    thetas = result_ll[:,0]
    train_mis = result_ll[:,1]
    true_mis = result_ll[:,2]

    plt.plot(thetas, train_mis, label='train')
    plt.plot(thetas, true_mis, label='true')
    plt.legend()
    plt.show()


def main():

    #****************************************#
    #    runtime params
    #****************************************#

    batch_size = 256
    B_N = 1
    nb_epochs = 20
    lr = 1e-3

    # wandb
    wandb.login()
    wandb.init(project="minfnet")

    #****************************************#
    #               build model 
    #****************************************#

    # create model
    model = modl.MI_Model(B_N=B_N, ctxt_N=1)
    model.to(rtut.device)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    #****************************************#
    #               load data 
    #****************************************#
    N_per_theta = int(3e4)

    thetas = np.linspace(0.2,2.5,7)
    random.shuffle(thetas)

    result_ll = []

    for theta in thetas:
        
        A_train, B_train, theta_train, *_ = inge.generate_noisy_channel_samples(N=N_per_theta, noise_std_nominal=theta, train_test_split=None)
        # import ipdb; ipdb.set_trace()
    
        dataset_train = dase.MinfDataset(A_var=A_train, B_var=B_train, thetas=theta_train)
        dataset_test = dase.MinfDataset(A_var=A_test, B_var=B_test, thetas=theta_test)

        train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

        #****************************************#
        #               train model
        #****************************************#

        train_acc_mi = modl.train(model, train_dataloader, nb_epochs, optimizer)
        train_true_mi = feature_selection.mutual_info_regression(A_train, B_train.ravel())[0]

        #****************************************#
        #              print results    
        # ****************************************#
         
        logger.info(f'theta {theta:.04f}: \t train MI {train_acc_mi:.04f} \t true MI {train_true_mi:.04f}')    
        result_ll.append([theta, train_acc_mi, train_true_mi])
        plot_results(result_ll)

    random.shuffle(thetas)
    N_per_theta = int(1e3)
    result_ll = []

    for theta in thetas:
        
        A_test, B_test, theta_test, *_ = inge.generate_noisy_channel_samples(N=N_per_theta, noise_std_nominal=theta, train_test_split=None)
        # import ipdb; ipdb.set_trace()
    
        dataset_test = dase.MinfDataset(A_var=A_test, B_var=B_test, thetas=theta_test)

        test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

        #****************************************#
        #               train model
        #****************************************#

        test_acc_mi = modl.test(model, test_dataloader)
        test_true_mi = feature_selection.mutual_info_regression(A_train, B_train.ravel())[0]

        #****************************************#
        #              print results    
        # ****************************************#
         
        logger.info(f'theta {theta:.04f}: \t test MI {test_acc_mi:.04f} \t true MI {test_true_mi:.04f}')    
        result_ll.append([theta, test_acc_mi, test_true_mi])
        plot_results(result_ll)


if __name__ == "__main__":
    main()