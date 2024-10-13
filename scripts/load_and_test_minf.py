import torch
import numpy as np
import matplotlib.pyplot as plt
from heputl import logging as heplog
import os

from minfnet.dats import datasets as dase
from minfnet.dats import input_generator as inge
import yaml
import minfnet.util.runtime_util as rtut
from minfnet.ml import mime_cond as modl
from sklearn import feature_selection




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



exp_dir = '/afs/cern.ch/user/k/kiwoznia/opde/opDeDe/results/noisy_channel_test/20240821_run5'
minf = torch.load(exp_dir+'/disc_model20240821_run5.pt')

with open(exp_dir+'/config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


logger.info('loading data from ' + exp_dir)
result_ll = np.load(exp_dir+'/result_ll.npz')

# create surrogate dataset
dataset = dase.SurrDataset(result_ll['theta'], result_ll['mi'])

N_per_theta = int(1e5)
result_ll = []

data_dict = {'A_test': [], 'B_test': [], 'theta_test': []}

for theta in dataset.theta:

    logger.info(f'generating data for theta {theta.item():.03f}')

    if config['theta_type'] == 'noise':
        A_test, B_test, theta_test, *_ = inge.generate_noisy_channel_samples(N=N_per_theta, noise_std_nominal=theta.item(), train_test_split=None)
    else:
        A_test, B_test, *_ = inge.generate_random_variables(N=N_per_theta, corr=theta.item(), train_test_split=None)
        theta_test = np.random.normal(loc=theta, scale=0.1, size=A_test.shape)

    data_dict['A_test'].append(A_test)
    data_dict['B_test'].append(B_test)
    data_dict['theta_test'].append(theta_test)

    dataset_test = dase.MinfDataset(A_var=A_test, B_var=B_test, thetas=theta_test)
    test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=config['batch_size'], shuffle=False)
    #****************************************#
    #               train model
    #****************************************#
    test_acc_mi = modl.test(minf, test_dataloader)
    test_true_mi = feature_selection.mutual_info_regression(A_test.reshape(-1,1), B_test)[0]
    #****************************************#
    #              print results    
    # ****************************************#
    logger.info(f'theta {theta.item():.03f}: \t test MI {test_acc_mi:.04f} \t true MI {test_true_mi:.04f}')    
    result_ll.append([theta.item(), test_acc_mi, test_true_mi])

plot_inputs(data_dict['A_test'], data_dict['B_test'], dataset.theta.numpy().squeeze(), plot_name='scatter_plot_inputs_test_from_loaded_model.png', fig_dir=exp_dir)
plot_results(result_ll,plot_name='mi_vs_theta_test_from_loaded_model.png',fig_dir=exp_dir)