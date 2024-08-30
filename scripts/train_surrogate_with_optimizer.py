import datetime
from minfnet.ml import surrogate as surr
from minfnet.ml import optimizer as optm
import torch
import matplotlib.pyplot as plt
import numpy as np
from minfnet.dats import datasets as dase
import torch.nn as nn
import torch.optim as optim

import argparse  # Import the argparse module
from heputl import logging as heplog

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


parser = argparse.ArgumentParser()
parser.add_argument('--run_n', type=int, default=0, help='Run number')
parser.add_argument('-c', '--choice', dest='thetaT', type=str, default='corr', choices=['noise', 'corr'], help='Choice between "noise" or "corr"')
args = parser.parse_args()

#****************************************************************#
#                       set up surrogate model
#****************************************************************#

logger.info('setting up surrogate model')

surrogate = surr.MLP_Surrogate(N_feat=2)

metric = nn.MSELoss()
optimizer = optim.Adam(surrogate.parameters(), lr=0.05)


#****************************************************************#
#                       load data
#****************************************************************#

# load minf results
exp_dir = '/afs/cern.ch/user/k/kiwoznia/opde/opDeDe/results/noisy_channel_test/20240829_run13'
logger.info('loading data from ' + exp_dir)
result_ll = np.load(exp_dir+'/result_ll_train.npz')

# create surrogate dataset
thetas = np.column_stack((result_ll['theta1'], result_ll['theta2']))
dataset = dase.SurrDataset(thetas, result_ll['mi'])


plot_theta_vs_mi(result_ll['theta1'], result_ll['mi'], plot_name='theta1_vs_mi', fig_dir=exp_dir)
plot_theta_vs_mi(result_ll['theta2'], result_ll['mi'], plot_name='theta2_vs_mi', fig_dir=exp_dir)

# define data loader
batch_size = 256
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#****************************************************************#
#                       train surrogate model
#****************************************************************#

logger.info('training surrogate model')

n_epochs = 500
for epoch in range(n_epochs):
    for i, batch in enumerate(dataloader):
        theta, mi = batch
        optimizer.zero_grad()
        output = surrogate(theta)
        loss = metric(output, mi)
        loss.backward()
        optimizer.step()
        logger.info(f'epoch {epoch}, batch {i}, loss {loss.item()}')

#****************************************************************#
#                       find theta with best MI
#****************************************************************#

surr_optimizer = optm.Optimizer(surr_dataset=dataset, surrogate=surrogate, epoch_n=300)
thetas = surr_optimizer.optimize()

# get mutual information for thetas

thetas = torch.tensor(thetas)
mis = surrogate(thetas)

thetas_np = thetas.detach().numpy().squeeze()
mis_np = mis.detach().numpy().squeeze()

plot_theta_vs_mi(thetas_np[:,0], mis_np, scatter_thetas=True, plot_name='theta1_vs_mi_descent', fig_dir=exp_dir)
plot_theta_vs_mi(thetas_np[:,1], mis_np, scatter_thetas=True, plot_name='theta2_vs_mi_descent', fig_dir=exp_dir)
