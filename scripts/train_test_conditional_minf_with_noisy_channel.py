import argparse
import datetime
import os
import numpy as np
import torch
from sklearn import feature_selection
import wandb

from minfnet.dats import input_generator as inge
from minfnet.dats import datasets as dase
from minfnet.ml import mime as modl
from minfnet.util import runtime_util as rtut
from minfnet.util import string_constants as stco

from heputl import logging as heplog
import random
import matplotlib.pyplot as plt


logger = heplog.get_logger(__name__)


def plot_inputs(A_list, B_list, theta_list, plot_name='scatter_plot.png', fig_dir='results'):
    num_plots = len(A_list)
    fig, axs = plt.subplots(1, num_plots, figsize=(6*num_plots,8))
    for i in range(num_plots):
        axs[i].scatter(A_list[i], B_list[i])
        axs[i].set_xlabel('A')
        axs[i].set_ylabel('B')
        axs[i].set_title(f'theta={theta_list[i]:.03f}')
    plot_path = os.path.join(fig_dir, plot_name)
    logger.info(f'saving plot to {plot_path}')
    plt.savefig(plot_path)
    plt.show()

def plot_histogram(thetas, thetas_nominal, plot_name='theta_histogram.png', fig_dir='results'):
    num_plots = len(thetas)
    fig, axs = plt.subplots(1, num_plots, figsize=(6*num_plots,8))
    for i, (theta, theta_nominal) in enumerate(zip(thetas, thetas_nominal)):
        axs[i].hist(theta, bins=50, alpha=0.5, label='theta distribution')
        axs[i].axvline(x=theta_nominal, color='red', linestyle='--', label=f'theta={theta_nominal:.03f}')
        axs[i].set_xlabel('Theta')
        axs[i].set_ylabel('Frequency')
        axs[i].legend()
    plot_path = os.path.join(fig_dir, plot_name)
    logger.info(f'saving histogram plot to {plot_path}')
    plt.savefig(plot_path)
    plt.show()



def plot_results(result_ll,plot_name='mi_vs_theta.png',fig_dir='results.png'):

    result_ll = np.array(result_ll)
    result_ll = result_ll[result_ll[:, 0].argsort()]
    thetas = result_ll[:,0]
    train_mis = result_ll[:,1]
    true_mis = result_ll[:,2]

    plt.figure(figsize=(8,6))
    plt.plot(thetas, train_mis, label='approx mi')
    plt.plot(thetas, true_mis, label='true mi')
    plt.legend()
    plt.xlabel('Theta/noise level')
    plt.ylabel('Mutual Information')
    plot_path = os.path.join(fig_dir,plot_name)
    logger.info(f'saving plot to {plot_path}')
    plt.savefig(plot_path)
    plt.show()


def main():

    #****************************************#
    #    runtime params
    #****************************************#

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run', dest='run_n', type=int, default=0, help='Run number')
    parser.add_argument('-c', '--choice', dest='thetaT', type=str, default='corr', choices=['noise', 'corr'], help='Choice between "noise" or "corr"')
    args = parser.parse_args()

    batch_size = 256
    B_N = 1
    nb_epochs = 50
    lr = 1e-3
    datestr = datetime.datetime.now().strftime('%Y%m%d') + '_run' + str(args.run_n)
    fig_dir = '/afs/cern.ch/user/k/kiwoznia/opde/opDeDe/results/noisy_channel_test/' + datestr
    os.makedirs(fig_dir, exist_ok=True)


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
    N_per_theta = int(5e4)

    thetas = np.linspace(0.1,2.3,7) if args.thetaT == 'noise' else list(stco.configs_random.values())
    random.shuffle(thetas)

    result_ll = []

    data_dict = {'A_train': [], 'B_train': [], 'theta_train': []}

    for theta in thetas:

        logger.info(f'generating data for theta {theta:.03f}')

        if args.thetaT == 'noise':
            A_train, B_train, theta_train, *_ = inge.generate_noisy_channel_samples(N=N_per_theta, noise_std_nominal=theta, train_test_split=None)
        else:
            A_train, B_train, *_ = inge.generate_random_variables(N=N_per_theta, corr=theta, train_test_split=None)
            theta_train = np.random.normal(loc=theta, scale=0.1, size=A_train.shape)

        data_dict['A_train'].append(A_train)
        data_dict['B_train'].append(B_train)
        data_dict['theta_train'].append(theta_train)

        dataset_train = dase.MinfDataset(A_var=A_train, B_var=B_train, thetas=theta_train)
        train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

        #****************************************#
        #               train model
        #****************************************#
        train_acc_mi = modl.train(model, train_dataloader, nb_epochs, optimizer)
        train_true_mi = feature_selection.mutual_info_regression(A_train.reshape(-1,1), B_train)[0]

        #****************************************#
        #              print results    
        # ****************************************#
        logger.info(f'theta {theta:.03f}: \t train MI {train_acc_mi:.04f} \t true MI {train_true_mi:.04f}')    
        result_ll.append([theta, train_acc_mi, train_true_mi])
    
    plot_inputs(data_dict['A_train'], data_dict['B_train'], thetas, plot_name='scatter_plot_inputs_train.png', fig_dir=fig_dir)
    plot_results(result_ll,plot_name='mi_vs_theta_train.png',fig_dir=fig_dir)
    plot_histogram(data_dict['theta_train'], thetas, plot_name='theta_train_histogram.png', fig_dir=fig_dir)


    model_path = fig_dir+'/disc_model'+datestr+'.pt'
    logger.info('saving model to ' + model_path)
    torch.save(model, model_path)


    random.shuffle(thetas)
    N_per_theta = int(5e4)
    result_ll = []

    data_dict = {'A_test': [], 'B_test': [], 'theta_test': []}

    for theta in thetas:
    
        logger.info(f'generating data for theta {theta:.03f}')

        if args.thetaT == 'noise':
            A_test, B_test, theta_test, *_ = inge.generate_noisy_channel_samples(N=N_per_theta, noise_std_nominal=theta, train_test_split=None)
        else:
            A_test, B_test, *_ = inge.generate_random_variables(N=N_per_theta, corr=theta, train_test_split=None)
            theta_test = np.random.normal(loc=theta, scale=0.1, size=A_test.shape)

        data_dict['A_test'].append(A_test)
        data_dict['B_test'].append(B_test)
        data_dict['theta_test'].append(theta_test)
    
        dataset_test = dase.MinfDataset(A_var=A_test, B_var=B_test, thetas=theta_test)
        test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
        #****************************************#
        #               train model
        #****************************************#
        test_acc_mi = modl.test(model, test_dataloader)
        test_true_mi = feature_selection.mutual_info_regression(A_test.reshape(-1,1), B_test)[0]
        #****************************************#
        #              print results    
        # ****************************************#
        logger.info(f'theta {theta:.03f}: \t test MI {test_acc_mi:.04f} \t true MI {test_true_mi:.04f}')    
        result_ll.append([theta, test_acc_mi, test_true_mi])
    
    plot_inputs(data_dict['A_test'], data_dict['B_test'], thetas, plot_name='scatter_plot_inputs_test.png', fig_dir=fig_dir)
    plot_results(result_ll,plot_name='mi_vs_theta_test.png',fig_dir=fig_dir)
    plot_histogram(data_dict['theta_test'], thetas, plot_name='theta_test_histogram.png', fig_dir=fig_dir)

    result_ll = np.array(result_ll)
    np.savez(os.path.join(fig_dir, 'result_ll.npz'), theta=result_ll[:, 0], mi=result_ll[:, 1])


if __name__ == "__main__":
    main()