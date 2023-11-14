import sys
import pandas as pd
import numpy as np
import math
from collections import namedtuple
import argparse

import torch, torchvision
from torch import nn
import torch.nn.functional as F

import src.input_generator as inge
import src.util as uti
import src.string_constants as stco


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
    def __init__(self, B_N=8):
        super(MI_Model, self).__init__()
        self.features_a = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 128), nn.ReLU(),
        )

        self.features_b = nn.Sequential(
            nn.Linear(B_N, 32), nn.ReLU(),
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

    entropy = 0 #compute_entropy(input_b) -> check how to compute entropy values for sanity check (not a binary array anymore)
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

    return acc_mi


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

    return test_acc_mi



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='read arguments for k-fold QR training')
    parser.add_argument('-n', dest='N', type=int, help='number of samples', default=int(1e5))
    parser.add_argument('-in', dest='input_type', choices=['calo', 'sensor', 'random'], help='type of inputs: calorimeter read from file, toy sensors or random variables', default='random')

    args = parser.parse_args()


    for config_name, file_path in stco.configs_calo.items():

        print(f'running train and test for {config_name}')

        # create training and test inputs
        A_train, B_train, A_test, B_test = inge.read_inputs_from_file(file_path, b_label='sensor_energy')
        # A_train, B_train, A_test, B_test = inge.generate_inputs(samples_N=samples_N)
        B_N = B_train.size(1)

        # import ipdb; ipdb.set_trace()

        # runtime params
        batch_size = A_train.size(0) if A_train.size(0) < 256 else 256
        nb_epochs = 10

        # create model
        model = MI_Model(B_N=B_N)
        model.to(uti.device)

        # train model
        train_acc_mi = train(model, A_train, B_train, batch_size, nb_epochs)
        print(f'train {config_name}: {train_acc_mi:.04f}') # {entropy(classes) / math.log(2):.04f}')

        # test model
        test_acc_mi = test(model, A_test, B_test, batch_size)
        print(f'test {config_name}: {test_acc_mi:.04f}') # {entropy(classes) / math.log(2):.04f}')






