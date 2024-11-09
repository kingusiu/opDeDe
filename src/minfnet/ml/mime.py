import math
import sys
import torch
from torch import nn
import torch.nn.functional as F
import wandb
import minfnet.util.runtime_util as rtut

from heputl import logging as heplog

logger = heplog.get_logger(__name__)

##################################
#               model
##################################

acti_dd = { 'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(), 'elu': nn.ELU() , 'leaky': nn.LeakyReLU() }


class MI_Model(nn.Module):

    def __init__(self, B_N, A_N=1, encoder_N=128, acti='relu', acti_out=None):

        super(MI_Model, self).__init__()

        # encoder for variable of interest / target (e.g. true energy)
        self.features_a = nn.Sequential(
            nn.Linear(A_N, 32), acti_dd[acti],
            nn.Linear(32, 32), acti_dd[acti],
            nn.Linear(32, encoder_N), acti_dd[acti],
        )

        # encoder for informing variables (e.g. calo hits)
        self.features_b = nn.Sequential(
            nn.Linear(B_N, 32), acti_dd[acti],
            nn.Linear(32, 32), acti_dd[acti],
            nn.Linear(32, encoder_N), acti_dd[acti],
        )

        connected_mlp = []
        connected_mlp.append(nn.Linear(encoder_N*2, 200))
        connected_mlp.append(acti_dd[acti])
        connected_mlp.append(nn.Linear(200, 1))
        if acti_out is not None:
            connected_mlp.append(acti_dd[acti_out])

        self.fully_connected = nn.Sequential(*connected_mlp)

    def forward(self, a, b):
        a = self.features_a(a).view(a.size(0), -1)
        b = self.features_b(b).view(b.size(0), -1)
        x = torch.cat((a, b), 1) # first dimension is batch-dimension
        return self.fully_connected(x)



def mutual_info(dep_ab, indep_ab, eps=1e-8):

    return dep_ab.mean() - torch.log(indep_ab.exp().mean()+eps) # means over batch

def get_dep_indep_batch(batch):

    batch_a, batch_b, batch_br = [b.to(rtut.device) for b in batch]

    return (batch_a, batch_b), (batch_a, batch_br)

def train(model: MI_Model, dataloader, nb_epochs, optimizer, eps=1e-8, wandb_log=False):

    model.train()

    train_mi = []

    for e in range(nb_epochs):
        
        acc_mi = 0.0
        
        for batch in dataloader:
            
            dep_batch, indep_batch = get_dep_indep_batch(batch)

            # apply the model: pass a & b and a & b_permuted
            dep_ab = model(*dep_batch)
            indep_ab = model(*indep_batch)

            mi = mutual_info(dep_ab=dep_ab, indep_ab=indep_ab, eps=eps)
            loss = -mi
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            acc_mi += mi.item()
        
        acc_mi /= len(dataloader)  # mi per batch
        acc_mi /= math.log(2)
        
        if wandb_log:
            wandb.log({"epoch": e, "mi": acc_mi})
        elif e % 20 == 0:
            logger.info(f'epoch {e}: mi {acc_mi:.04f}')
        train_mi.append(acc_mi) # train-mi per epoch

    return acc_mi


def test(model, dataloader, eps=1e-8):

    model.eval()
    test_acc_mi = 0.0

    for batch in dataloader:

        batch_a, batch_b, batch_br, _ = [b.to(rtut.device) for b in batch]

        dep_ab = model(batch_a, batch_b)
        indep_ab = model(batch_a, batch_br)
        
        mi = mutual_info(dep_ab=dep_ab, indep_ab=indep_ab, eps=eps)
        test_acc_mi += mi.item()

    test_acc_mi /= len(dataloader)
    test_acc_mi /= math.log(2)

    return test_acc_mi
