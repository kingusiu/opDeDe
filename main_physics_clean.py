#!/usr/bin/env python

#########################################################################
# This program is free software: you can redistribute it and/or modify  #
# it under the terms of the version 3 of the GNU General Public License #
# as published by the Free Software Foundation.                         #
#                                                                       #
# This program is distributed in the hope that it will be useful, but   #
# WITHOUT ANY WARRANTY; without even the implied warranty of            #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU      #
# General Public License for more details.                              #
#                                                                       #
# You should have received a copy of the GNU General Public License     #
# along with this program. If not, see <http://www.gnu.org/licenses/>.  #
#                                                                       #
# Written by Francois Fleuret, (C) Idiap Research Institute             #
#                                                                       #
# Contact <francois.fleuret@idiap.ch> for comments & bug reports        #
#########################################################################

import argparse, math, sys
from copy import deepcopy

import torch, torchvision
import matplotlib.pyplot as plt

from torch import nn
import torch.nn.functional as F
import os
import json

NB_SENSORS = 8

######################################################################

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

######################################################################

parser = argparse.ArgumentParser(
    description = '''An implementation of a Mutual Information estimator with a deep model

    Three different toy data-sets are implemented, each consists of
    pairs of samples, that may be from different spaces:

    (1) Two MNIST images of same class. The "true" MI is the log of the
    number of used MNIST classes.

    (2) One MNIST image and a pair of real numbers whose difference is
    the class of the image. The "true" MI is the log of the number of
    used MNIST classes.

    (3) Two 1d sequences, the first with a single peak, the second with
    two peaks, and the height of the peak in the first is the
    difference of timing of the peaks in the second. The "true" MI is
    the log of the number of possible peak heights.''',

    formatter_class = argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--data',
                    type = str, default = 'toy_dec_design',
                    help = 'What data: image_pair, image_values_pair, sequence_pair')

parser.add_argument('--seed',
                    type = int, default = 0,
                    help = 'Random seed (default 0, < 0 is no seeding)')

# parser.add_argument('--nb_sensors',
#                     type = int, default = 16,
#                     help = 'How many sensors')

parser.add_argument('--nb_train',
                    type = int, default = 1000000,
                    help = 'How many samples for training')

parser.add_argument('--nb_test',
                    type = int, default = 50000,
                    help = 'How many samples for testing')

parser.add_argument('--nb_epochs',
                    type = int, default = 5,
                    help = 'How many epochs')

parser.add_argument('--batch_size',
                    type = int, default = 512,
                    help = 'Batch size')

parser.add_argument('--learning_rate',
                    type = float, default = 1e-3,
                    help = 'Batch size')

# parser.add_argument('--independent', action = 'store_true',
#                     help = 'Should the pair components be independent')

######################################################################

args = parser.parse_args()

if args.seed >= 0:
    torch.manual_seed(args.seed)

######################################################################

def entropy(target):
    probas = []
    for k in range(target.max() + 1):
        n = (target == k).sum().item()
        if n > 0: probas.append(n)
    probas = torch.tensor(probas).float()
    probas /= probas.sum()
    return - (probas * probas.log()).sum().item()

######################################################################

# def generate(
#         nb,
#         r_sensor=1e-2,
#         nb_sensors=16,
#         epsilon=math.pi/10,
#         single_configuration=True
# ):
#     alpha = torch.rand(nb) * 2 * math.pi
#     beta = alpha + (torch.rand(nb) - 0.5) * epsilon
#     x=torch.rand(1 if single_configuration else nb, nb_sensors)*2-1
#     y=torch.rand(1 if single_configuration else nb, nb_sensors)*2-1
#     beta=beta[:,None]
#     z=((-beta.sin() * x + beta.cos() * y).abs() <= r_sensor).float()
#     valid = (torch.arccos((x*beta.cos() + y*beta.sin()) / torch.sqrt(x**2 + y**2)) <= math.pi/2)
#     z = z * valid

#     return alpha.unsqueeze(-1).to(device), z.to(device), None

# alpha, z = generate(10)
# print(f"{alpha.size()=} {z.size()=}")

def generate(
        nb,
        r_sensor=0.05,
        nb_sensors=16,
        epsilon=math.pi/10,
        sensor_config=True,
        single_configuration=True
):
    alpha = torch.rand(nb) * 2 * math.pi
    beta = alpha + (torch.rand(nb) - 0.5) * epsilon

    if sensor_config == 'random':
        ## Randomly placed sensors
        x=torch.rand(1 if single_configuration else nb, nb_sensors)*2-1
        y=torch.rand(1 if single_configuration else nb, nb_sensors)*2-1

        # x = torch.tensor([[ 0.3383,  0.6828, -0.8881, -0.1610,  0.1105, -0.5459, -0.1385, -0.6813,
        #     0.8355,  0.2302,  0.0901,  0.0475,  0.7599, -0.6489, -0.6065, -0.6200]])
        
        # y = torch.tensor([[-0.5504, -0.9351, -0.0326, -0.8577,  0.4813, -0.9587, -0.2622, -0.8422,
        #     0.2316,  0.7447, -0.1618, -0.4690,  0.4523,  0.8592,  0.8261, -0.0340]])

    elif sensor_config == 'ring':
        # Sensors placed on a circle
        theta = 2*math.pi/nb_sensors
        x = 0.5*torch.cos(torch.arange(0, nb_sensors) * theta)
        y = 0.5*torch.sin(torch.arange(0, nb_sensors) * theta)

    elif sensor_config == 'linear':
        # Sensors placed on a line
        theta = math.pi/4
        x = (0.1 + torch.arange(0, nb_sensors)) * torch.cos(torch.tensor(theta)) * r_sensor * 2
        y = x

    
    beta=beta[:,None]
    z=((-beta.sin() * x + beta.cos() * y).abs() <= r_sensor).float()

    valid = (((x*beta.cos() + y*beta.sin()) / torch.sqrt(x**2 + y**2)) >= 0)
    z = z * valid

    return alpha.unsqueeze(-1).to(device), z.to(device), None


######################################################################

class NetForInitialConditionSensorOutputPair(nn.Module):
    def __init__(self, nb_sensors=16):
        super(NetForInitialConditionSensorOutputPair, self).__init__()
        self.features_a = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 128), nn.ReLU(),
        )

        self.features_b = nn.Sequential(
            nn.Linear(nb_sensors, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 128), nn.ReLU(),
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(256, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def forward(self, a, b):
        a = self.features_a(a).view(a.size(0), -1)
        b = self.features_b(b).view(b.size(0), -1)
        x = torch.cat((a, b), 1)
        return self.fully_connected(x)

######################################################################

train_mis = []
test_mis = []

# each tuple respresents (r_sensor, nb_sensors, epsilon, sensor_config, plot-tag)
eps = 0.
configs = [
            #(0.05, 16, eps, 'random', 'random'),
          #(0.05, 16, eps, 'ring', 'ring (small)'),
          (0.09, NB_SENSORS, eps, 'ring', 'ring (large)'),
          #(0.0145, 16, eps, 'linear', 'linear')
          ]

# configs = [(0.05, 16, 0, 'random', 'random'),
#           (0.05, 16, 0, 'ring', 'ring (small)'),
#           (0.097, 16, 0, 'ring', 'ring (large)'),
#           (0.0145, 16, 0, 'linear', 'linear')]

# r_sensors = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11]
# configs = [0.01, 16, 0., 'ring', 'ring']
# configs = [(r, configs[1], configs[2], configs[3], f'{configs[4]} (r={r})') for r in r_sensors]

# epsilons = [math.pi*x for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]]
# configs = [0.05, 16, math.pi/10, 'ring', 'ring']
# configs = [(configs[0], configs[1], eps, configs[3], f'{configs[4]} (eps={eps:.3f})') for eps in epsilons]

# check for the last folder names configvN in a given directory
def get_last_config_number(path):
    try:
        last_version =  max([int(f.split('configv')[-1]) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])
        return f'configv{last_version+1}'
    except:
        return 'configv1'
    
path = 'results'
results_path = os.path.join(path, get_last_config_number(path))
os.makedirs(results_path)

with open(os.path.join(results_path, 'config.txt'), 'w') as f:
    json.dump(configs, f)

# r_sensor = 0.05
# epsilon = math.pi/10
# r_sensors = torch.arange(0.02, 0.12, 0.01)
# epsilons =  math.pi*torch.arange(0.1, 1.1, 0.1)

common_fig, common_ax = plt.subplots()
colors = ['r', 'b', 'g', 'y', 'k', 'm', 'c', 'orange', 'purple', 'brown']

def compute_entropy(z):
    # z = torch.randint(2, (10000,7))
    assert tuple(z.unique()) == (0,1)
    pow2=(2**torch.arange(z.size(1)))[None,:].to(z.device)
    print(f'SANITY max {z.sum(1).max()}, min {z.sum(1).min()}')

    c=0
    for zz in z.split(1000):
        n=(zz.long()*pow2).sum(1)
        c+=torch.nn.functional.one_hot(n).sum(0)

    # c=torch.nn.functional.one_hot(n).sum(0)
    p=c/c.sum()
    print(f'SANITY  p {p}')
    h=-p.xlogy(p).sum() / math.log(2)

    return h.item()

# here starts main (probably)

# for each of the detector configurations
for idx, config in enumerate(configs):
    print(f'Config : r_sensor={config[0]}, nb_sensors={config[1]}, epsilon={config[2]}, sensor_config={config[3]}')
    

    if args.data == 'toy_dec_design':
        create_pairs = generate
        model = NetForInitialConditionSensorOutputPair(nb_sensors=NB_SENSORS)


        # ######################
        # ## Save for figures
        # a, b, c = create_pairs()
        # for k in range(10):
        #     file = open(f'train_{k:02d}.dat', 'w')
        #     for i in range(a.size(1)):
        #         file.write(f'{a[k, i]:f} {b[k,i]:f}\n')
        #     file.close()
        # ######################

    else:
        raise Exception('Unknown data ' + args.data)

    ######################################################################
    # Train

    print(f'nb_parameters {sum(x.numel() for x in model.parameters())}')

    model.to(device)


    input_a, input_b, classes = create_pairs(nb=args.nb_train, 
                                             r_sensor=config[0], 
                                             nb_sensors=config[1], 
                                             epsilon=config[2],
                                             sensor_config=config[3])
    
    entropy = compute_entropy(input_b)
    print(f'Entropy : {entropy:.3f}')

    train_mi = []

    for e in range(args.nb_epochs):

        optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)

        input_br = input_b[torch.randperm(input_b.size(0))]

        acc_mi = 0.0

        for batch_a, batch_b, batch_br in zip(input_a.split(args.batch_size),
                                            input_b.split(args.batch_size),
                                            input_br.split(args.batch_size)):
            mi = model(batch_a, batch_b).mean() - model(batch_a, batch_br).exp().mean().log()
            acc_mi += mi.item()
            loss = - mi
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc_mi /= (input_a.size(0) // args.batch_size)
        acc_mi /= math.log(2)

        train_mi.append(acc_mi)

        print(f'{e+1} {acc_mi:.04f}') # {entropy(classes) / math.log(2):.04f}')

        sys.stdout.flush()

    ######################################################################
    # Test

    input_a, input_b, classes = create_pairs(nb=args.nb_train, 
                                             r_sensor=config[0], 
                                             nb_sensors=config[1], 
                                             epsilon=config[2],
                                             sensor_config=config[3])
    
    input_br = input_b[torch.randperm(input_b.size(0))]

    test_acc_mi = 0.0

    for batch_a, batch_b, batch_br in zip(input_a.split(args.batch_size),
                                        input_b.split(args.batch_size),
                                        input_br.split(args.batch_size)):
        mi = model(batch_a, batch_b).mean() - model(batch_a, batch_br).exp().mean().log()
        test_acc_mi += mi.item()

    test_acc_mi /= (input_a.size(0) // args.batch_size)
    test_acc_mi /= math.log(2)
    print(f'test {test_acc_mi:.04f}') # {entropy(classes) / math.log(2):.04f}')

    ######################################################################

    fig, ax = plt.subplots()
    ax.plot(train_mi, label='train')
    ax.axhline(y=test_acc_mi, color='r', label='test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MI')
    ax.legend(loc='best')
    fig.savefig(os.path.join(results_path, f'{config[-1]}.png'))

    common_ax.plot(train_mi, label=config[-1], color=colors[idx])
    common_ax.axhline(y=test_acc_mi, color=colors[idx])
    common_ax.set_xlabel('Epoch')
    common_ax.set_ylabel('MI')
    common_ax.legend(loc='best')

    train_mis.append(sum(train_mi[-50:])/50)
    test_mis.append(test_acc_mi)

common_ax.set_ylim([0, 4.])
common_fig.savefig(os.path.join(results_path, f'all.png'))

# fig, ax = plt.subplots()
# ax.plot(r_sensors, train_mis, label='train')
# ax.plot(r_sensors, test_mis, label='test')
# ax.set_xlabel('Sensor Radius')
# ax.set_ylabel('MI')
# ax.legend(loc='best')
# fig.savefig(os.path.join(results_path, f'resultRs.png'))

# fig, ax = plt.subplots()
# ax.plot(epsilons, train_mis, label='train')
# ax.plot(epsilons, test_mis, label='test')
# ax.set_xlabel('Epsilon')
# ax.set_ylabel('MI')
# ax.legend(loc='best')
# fig.savefig(os.path.join(results_path, f'resultEs.png'))


