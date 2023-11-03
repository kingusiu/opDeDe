import sys
import pandas as pd
import numpy as np
import math
from collections import namedtuple

import torch, torchvision

from torch import nn
import torch.nn.functional as F

import src.input_generator as inge
import src.util as uti


##################################
#               entropy
##################################


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



##################################
#               model
##################################

class MI_Model(nn.Module):
    def __init__(self, nb_sensors=8):
        super(MI_Model, self).__init__()
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


def train(model,input_a,input_b,batch_size,nb_epochs): 

    entropy = compute_entropy(input_b)
    print(f'Entropy : {entropy:.3f}')

    train_mi = []

    learning_rate = 1e-3

    for e in range(nb_epochs):

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        input_br = input_b[torch.randperm(input_b.size(0))]

        acc_mi = 0.0

        for batch_a, batch_b, batch_br in zip(input_a.split(batch_size),
                                            input_b.split(batch_size),
                                            input_br.split(batch_size)):
            mi = model(batch_a, batch_b).mean() - model(batch_a, batch_br).exp().mean().log()
            acc_mi += mi.item()
            loss = - mi
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc_mi /= (input_a.size(0) // batch_size)
        acc_mi /= math.log(2)

        train_mi.append(acc_mi)

        print(f'{e+1} {acc_mi:.04f}') # {entropy(classes) / math.log(2):.04f}')

        sys.stdout.flush()


def test(model, input_a, input_b, batch_size):

    input_br = input_b[torch.randperm(input_b.size(0))]

    test_acc_mi = 0.0

    for batch_a, batch_b, batch_br in zip(input_a.split(batch_size),
                                        input_b.split(batch_size),
                                        input_br.split(batch_size)):
        mi = model(batch_a, batch_b).mean() - model(batch_a, batch_br).exp().mean().log()
        test_acc_mi += mi.item()

    test_acc_mi /= (input_a.size(0) // batch_size)
    test_acc_mi /= math.log(2)
    print(f'test {test_acc_mi:.04f}') # {entropy(classes) / math.log(2):.04f}')



if __name__ == '__main__':

    # create training and test inputs
    config = [0.09, 8, 0., 'ring', 'ring (large)']
    alpha_train, hits_train = inge.generate(nb=int(1e5),r_sensor=config[0],nb_sensors=config[1],epsilon=config[2],sensor_config=config[3])
    alpha_test, hits_test = inge.generate(nb=int(1e5),r_sensor=config[0],nb_sensors=config[1],epsilon=config[2],sensor_config=config[3])

    # runtime params
    batch_size = 512
    nb_epochs = 15

    # create model
    model = MI_Model()
    model.to(uti.device)

    # train model
    train(model, alpha_train, hits_train, batch_size, nb_epochs)

    # test model
    test(model, alpha_test, hits_test, batch_size)






