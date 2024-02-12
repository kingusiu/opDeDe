import math
import sys
import torch, torchvision
from torch import nn
import torch.nn.functional as F


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
        x = torch.cat((a, b), 1) # first dimension is batch-dimension
        return self.fully_connected(x)


def train(model,input_a,input_b,batch_size,nb_epochs):

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
