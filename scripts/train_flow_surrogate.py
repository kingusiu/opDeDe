import datetime
from minfnet.ml import surrogate as surr
import torch
import matplotlib.pyplot as plt
import numpy as np
from minfnet.dats import datasets as dase
import torch.nn as nn
import torch.optim as optim
import sklearn as sk

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

tail_bound = 3.5
surrogate = surr.Flow_Surrogate(data_dim=1, context_dim=1, tail_bound=tail_bound)
# scale data to tailbound (3.5)!

metric = nn.MSELoss()
optimizer = optim.Adam(surrogate.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999)


#****************************************************************#
#                       load data
#****************************************************************#

# load minf results
exp_dir = '/afs/cern.ch/user/k/kiwoznia/opde/opDeDe/results/noisy_channel_test/20240815_run0'
logger.info('loading data from ' + exp_dir)
result_ll = np.load(exp_dir+'/result_ll.npz')

# create surrogate dataset
mi_targets = result_ll['mi']
mi_targets = sk.preprocessing.MinMaxScaler(feature_range=(-tail_bound, tail_bound)).fit_transform(\
    mi_targets[np.newaxis].T).astype(np.float32).squeeze()
dataset = dase.SurrDataset(result_ll['theta'], mi_targets)


plot_theta_vs_mi(result_ll['theta'], result_ll['mi'])

# define data loader
batch_size = 256
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#****************************************************************#
#                       train surrogate model
#****************************************************************#

logger.info('training surrogate model')

n_epochs = 300
losses = []

for epoch in range(n_epochs):

    loss_track = []

    for i, batch in enumerate(dataloader):
        theta, mi = batch
        optimizer.zero_grad()

        loss = -surrogate.log_prob(mi, context=theta).mean() # negative loss -> probs > 1
        loss_track.append(loss.cpu().detach().numpy())
        loss.backward()
        #clip_values = torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        scheduler.step()

    epoch_loss = np.float64(np.mean(loss_track))
    losses.append(epoch_loss)
    logger.info(f'epoch {epoch}, batch {i}, loss {epoch_loss.item()}')

#****************************************************************#
#                       find theta with best MI
#****************************************************************#

surrogate.eval()

theta = dataset.theta[np.random.randint(len(dataset.theta))]
theta.requires_grad = True
theta_bounds = (min(dataset.theta), max(dataset.theta))
step_sz = 8e-1
thetas = []
i = 0

while theta.item() > theta_bounds[0] and theta.item() < theta_bounds[1]:
    thetas.append(theta.item())
    logger.info(f'theta {i}: {theta.item():.3f}')
    mi = surrogate(theta)
    grad = torch.autograd.grad(outputs=mi, inputs=theta, retain_graph=True)[0]
    theta = theta + step_sz * grad # increase mi
    i += 1
    if abs(theta.item() - thetas[-1]) < 1e-4:
        break   

thetas = torch.tensor(thetas).reshape(-1, 1)
mis = surrogate(thetas)

thetas_np = thetas.detach().numpy().squeeze()
mis_np = mis.detach().numpy().squeeze()

plot_theta_vs_mi(thetas_np, mis_np, scatter_thetas=True, plot_name='theta_vs_mi_descent', fig_dir=exp_dir)
