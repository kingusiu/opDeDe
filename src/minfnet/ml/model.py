import math
import sys
import torch, torchvision
from torch import nn
import torch.nn.functional as F
import wandb


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


def mutual_info(model, batch_a, batch_b, batch_br, eps=1e-8):

    return model(batch_a, batch_b).mean() - torch.log(model(batch_a, batch_br).exp().mean()+eps)


def train(model,input_a,input_b,batch_size,nb_epochs,lr=1e-3,eps=1e-8):

    wandb.watch(model, mutual_info, log="all", log_freq=10)

    train_mi = []

    b_c = 0
    
    for e in range(nb_epochs):

        optimizer = torch.optim.Adam(model.parameters(), lr=lr) # put this outside of epoch loop!

        input_br = input_b[torch.randperm(input_b.size(0))]

        acc_mi = 0.0

        # import ipdb; ipdb.set_trace()

        for b_i, (batch_a, batch_b, batch_br) in enumerate(zip(input_a.split(batch_size),
                                            input_b.split(batch_size),
                                            input_br.split(batch_size))):

            mi = mutual_info(model, batch_a, batch_b, batch_br,eps)
            acc_mi += mi.item()
            loss = - mi
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ((b_i + 1) % 30) == 0:
                b_c += b_i
                wandb.log({"epoch": e, "loss": loss}, step=b_c)

        acc_mi /= (input_a.size(0) // batch_size) # mi per batch
        acc_mi /= math.log(2)

        wandb.log({"epoch": e, "mi": acc_mi})

        train_mi.append(acc_mi)

        print(f'{e+1} {acc_mi:.04f}') # {entropy(classes) / math.log(2):.04f}')

        sys.stdout.flush()

    return acc_mi


def test(model, input_a, input_b, batch_size,eps=1e-8):

    input_br = input_b[torch.randperm(input_b.size(0))]

    test_acc_mi = 0.0

    for batch_a, batch_b, batch_br in zip(input_a.split(batch_size),
                                        input_b.split(batch_size),
                                        input_br.split(batch_size)):

        mi = mutual_info(model, batch_a, batch_b, batch_br, eps)
        test_acc_mi += mi.item()

    test_acc_mi /= (input_a.size(0) // batch_size)
    test_acc_mi /= math.log(2)

    wandb.log({"test mi": test_acc_mi})

    return test_acc_mi
