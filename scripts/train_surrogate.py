import datetime
from minfnet.ml import surrogate as surr
import torch
import numpy as np
from minfnet.ml.dataset import SurrogateDataset
import torch.nn as nn
import torch.optim as optim

import argparse  # Import the argparse module
from heputl import logging as heplog

logger = heplog.get_logger(__name__)



parser = argparse.ArgumentParser()
parser.add_argument('--run_n', type=int, default=0, help='Run number')
args = parser.parse_args()

#****************************************************************#
#                       set up surrogate model
#****************************************************************#

logger.info('setting up surrogate model')

surrogate = surr.SurrogateModel()

metric = nn.MSELoss()
optimizer = optim.Adam(surrogate.parameters(), lr=0.0001)


#****************************************************************#
#                       load data
#****************************************************************#

# load minf results
exp_dir = '/afs/cern.ch/user/k/kiwoznia/opde/opDeDe/results/noisy_channel_test/20240815_run0'
logger.info('loading data from ' + exp_dir)
result_ll = np.load(exp_dir+'/result_ll.npz')

# create surrogate dataset
dataset = SurrogateDataset(result_ll['theta'], result_ll['mi'])

import matplotlib.pyplot as plt

# plot theta vs mi
plt.plot(dataset.theta, dataset.mi)
plt.xlabel('Theta')
plt.ylabel('MI')
plt.title('Theta vs MI')
plt.show()

# define data loader
batch_size = 256
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

