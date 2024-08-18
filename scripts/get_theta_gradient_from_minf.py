import torch
import numpy as np
import matplotlib.pyplot as plt
from heputl import logging as heplog

from minfnet.dats import datasets as dase


logger = heplog.get_logger(__name__)

def plot_theta_vs_mi(theta, mi, scatter_thetas=False, plot_name=None, fig_dir=None):
    # plot theta vs mi
    sorted_indices = np.argsort(theta)
    sorted_theta = theta[sorted_indices]
    sorted_mi = mi[sorted_indices]

    plt.plot(sorted_theta, sorted_mi)
    if scatter_thetas:
        plt.scatter(sorted_theta, sorted_mi, color='red', marker='>')
    plt.xlabel('Theta')
    plt.ylabel('MI')
    plt.title('Theta vs MI')
    if plot_name is not None and fig_dir is not None:
        plt.savefig(f'{fig_dir}/{plot_name}.png')
    plt.show()
    plt.close()


exp_dir = '/afs/cern.ch/user/k/kiwoznia/opde/opDeDe/results/noisy_channel_test/20240815_run0'
model = torch.load(exp_dir+'/disc_model20240815_run0.pt')

logger.info('loading data from ' + exp_dir)
result_ll = np.load(exp_dir+'/result_ll.npz')

# create surrogate dataset
dataset = dase.SurrDataset(result_ll['theta'], result_ll['mi'])


plot_theta_vs_mi(result_ll['theta'], result_ll['mi'])

# define data loader
batch_size = 256
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)