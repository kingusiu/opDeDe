#!/usr/bin/env python
# coding=utf-8

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
# Written by and Copyright (C) Francois Fleuret                         #
# Contact <francois.fleuret@idiap.ch> for comments & bug reports        #
#########################################################################

import math
from math import pi
import random

import numpy as np
import torch
import torchvision
from torch import nn, autograd
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib.collections as mc

def set_seed(seed):
    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(0)

######################################################################

def phi(x):
    p, std = 0.3, 0.2
    mu = (1 - p) * torch.exp(LogProba((x - 0.5) / std, math.log(1 / std))) + \
              p  * torch.exp(LogProba((x + 0.5) / std, math.log(1 / std)))
    return mu

def sample_phi(nb):
    p, std = 0.3, 0.2
    result = torch.empty(nb).normal_(0, std)
    result = result + torch.sign(torch.rand(result.size()) - p) / 2
    return result

######################################################################

# START_LOG_PROBA
def LogProba(x, ldj):
    log_p = ldj - 0.5 * (x**2 + math.log(2*pi))
    return log_p
# END_LOG_PROBA

######################################################################

# START_MODEL
class PiecewiseLinear(nn.Module):
    def __init__(self, nb, xmin, xmax):
        super().__init__()
        self.xmin = xmin
        self.xmax = xmax
        self.nb = nb
        self.alpha = nn.Parameter(torch.tensor([xmin], dtype = torch.float))
        mu = math.log((xmax - xmin) / nb)
        self.xi = nn.Parameter(torch.empty(nb + 1).normal_(mu, 1e-4))

    def forward(self, x):
        y = self.alpha + self.xi.exp().cumsum(0)
        u = self.nb * (x - self.xmin) / (self.xmax - self.xmin)
        n = u.long().clamp(0, self.nb - 1)
        a = (u - n).clamp(0, 1)
        x = (1 - a) * y[n] + a * y[n + 1]
        return x
# END_MODEL

    def invert(self, y):
        ys = self.alpha + self.xi.exp().cumsum(0).view(1, -1)
        yy = y.view(-1, 1)
        k = torch.arange(self.nb).view(1, -1)
        assert (y >= ys[0, 0]).min() and (y <= ys[0, self.nb]).min()
        yk = ys[:, :-1]
        ykp1 = ys[:, 1:]
        x = self.xmin + (self.xmax - self.xmin)/self.nb * ((yy >= yk) * (yy < ykp1).long() * (k + (yy - yk)/(ykp1 - yk))).sum(1)
        return x

class PositivePolynomial(nn.Module):
    def __init__(self, degree):
        super().__init__()
        a = torch.empty(degree).normal_(0, 1e-2)
        a[1] += 1.0
        self.a = nn.Parameter(a)

    def forward(self, x):
        r = self.a[1:].view(1, 1, -1)
        q = F.conv_transpose1d(r, r)
        d = torch.arange(q.size(2)) + 1.
        y = (x.view(-1, 1).pow(d) * (q / d).view(1, -1)).sum(1) + self.a[0]
        return y

class SumOfSigmoids(nn.Module):
    def __init__(self, nb, xmin, xmax):
        super().__init__()
        self.b = nn.Parameter(torch.linspace(xmin, xmax, nb).view(1, -1))
        self.nb = nb
        # self.d = torch.full((nb,), (xmax - xmin) / nb).log().view(1, -1)
        self.e = nn.Parameter(torch.full((nb,), 1e2).view(1, -1))
        self.alpha = nn.Parameter(torch.tensor([xmin, xmax], dtype = torch.float))

    def forward(self, x):
        y = torch.sigmoid((x.view(-1, 1) - self.b) * self.e).sum(1) * (self.alpha[1] - self.alpha[0]) / self.nb + self.alpha[0]
        return y

######################################################################

# input = torch.rand((100,))
# targets = (input/2 + 2 * torch.relu(input - 0.3333) - 2 * torch.relu(input - 0.66666)) * 0.8

# model = PositivePolynomial(8)

# optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)

# for k in range(10000):
    # loss = (model(input) - targets).pow(2).mean()
    # if k%100 == 0: print(k, loss.item())
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

# x = torch.linspace(0, 1, 100)
# y = model(x)

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.set_xlim(-0.1, 1.1)
# ax.set_ylim(-0.1, 1.1)

# ax.scatter(input, targets, color = 'blue', )
# ax.plot(x, y.detach(), color = 'red', )

# plt.show()

# exit(0)

######################################################################
# Training

nb_samples = 25000
nb_epochs = 250
batch_size = 100

model = PiecewiseLinear(nb = 1001, xmin = -4, xmax = 4)
# model = SumOfSigmoids(nb = 51, xmin = -4, xmax = 4)

# print(model(torch.linspace(-10, 10, 25)))

# exit(0)

# print('** TESTING WITH POSITIVE POLYNOMIAL!!!!')
# model = PositivePolynomial(degree = 16)

train_input = sample_phi(nb_samples)

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

for k in range(nb_epochs):
    acc_loss = 0

# START_OPTIMIZATION
    for input in train_input.split(batch_size):
        input.requires_grad_()
        output = model(input)

        derivatives, = autograd.grad(
            output.sum(), input,
            retain_graph = True, create_graph = True
        )

        loss = ( 0.5 * (output**2 + math.log(2*pi)) - derivatives.log() ).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# END_OPTIMIZATION

        acc_loss += loss.item()
    if k%10 == 0: print(k, loss.item())

######################################################################

input = torch.linspace(-3, 3, 175)

mu = phi(input)
mu_N = torch.exp(LogProba(input, 0))

input.requires_grad_()
output = model(input)

grad = autograd.grad(output.sum(), input)[0]
mu_hat = LogProba(output, grad.log()).detach().exp()

######################################################################
# FIGURES

result_path = '/afs/cern.ch/user/k/kiwoznia/opde/opDeDe/results/oneD_flow_test'


input = input.detach().numpy()
output = output.detach().numpy()
mu = mu.numpy()
mu_hat = mu_hat.numpy()

######################################################################

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# ax.set_xlim(-5, 5)
# ax.set_ylim(-5, 5)
# ax.set_aspect('equal')
# ax.axis('off')

ax.plot(input, output, '-', color = 'tab:red')

filename = result_path+'/miniflow_mapping.pdf'
print(f'Saving {filename}')
fig.savefig(filename, bbox_inches = 'tight')

# plt.show()

######################################################################

green_dist = '#bfdfbf'

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# ax.set_xlim(-4.5, 4.5)
# ax.set_ylim(-0.1, 1.1)
lines = list(([(x_in.item(), 0), (x_out.item(), 0.5)]) for (x_in, x_out) in zip(input, output))
lc = mc.LineCollection(lines, color = 'tab:red', linewidth = 0.1)
ax.add_collection(lc)
ax.axis('off')

ax.fill_between(input,  0.52, mu_N * 0.2 + 0.52, color = green_dist)
ax.fill_between(input, -0.30, mu   * 0.2 - 0.30, color = green_dist)

filename = result_path+'/miniflow_flow.pdf'
print(f'Saving {filename}')
fig.savefig(filename, bbox_inches = 'tight')

# plt.show()

######################################################################

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.axis('off')

ax.fill_between(input, 0, mu, color = green_dist)
# ax.plot(input, mu, '-', color = 'tab:blue')
# ax.step(input, mu_hat, '-', where = 'mid', color = 'tab:red')
ax.plot(input, mu_hat, '-', color = 'tab:red')

filename = result_path+'/miniflow_dist.pdf'
print(f'Saving {filename}')
fig.savefig(filename, bbox_inches = 'tight')

# plt.show()

######################################################################

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.axis('off')

# ax.plot(input, mu, '-', color = 'tab:blue')
ax.fill_between(input, 0, mu, color = green_dist)
# ax.step(input, mu_hat, '-', where = 'mid', color = 'tab:red')

filename = result_path+'/miniflow_target_dist.pdf'
print(f'Saving {filename}')
fig.savefig(filename, bbox_inches = 'tight')

# plt.show()

######################################################################

if hasattr(model, 'invert'):
    z = torch.randn(200)
    z = z[(z > -3) * (z < 3)]

    x = model.invert(z)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-0.1, 1.1)
    lines = list(([(x_in.item(), 0), (x_out.item(), 0.5)]) for (x_in, x_out) in zip(x, z))
    lc = mc.LineCollection(lines, color = 'tab:red', linewidth = 0.1)
    ax.add_collection(lc)
    # ax.axis('off')

    # ax.fill_between(input,  0.52, mu_N * 0.2 + 0.52, color = green_dist)
    # ax.fill_between(input, -0.30, mu   * 0.2 - 0.30, color = green_dist)

    filename = result_path+'/miniflow_synth.pdf'
    print(f'Saving {filename}')
    fig.savefig(filename, bbox_inches = 'tight')

    # plt.show()