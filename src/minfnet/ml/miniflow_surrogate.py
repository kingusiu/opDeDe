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
        u = self.nb * (x - self.xmin) / (self.xmax - self.xmin) # find index of the piece where x falls into
        n = u.long().clamp(0, self.nb - 1) # make sure index of piece is within number of pieces xi
        a = (u - n).clamp(0, 1) # find relative position of x within piece to use as weight of the two y anchor points
        x = (1 - a) * y[n] + a * y[n + 1] # linear interpolation between two neighboring pieces left and right of x
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
