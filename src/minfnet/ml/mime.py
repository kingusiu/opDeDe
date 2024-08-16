import math
import sys
import torch, torchvision
from torch import nn
import torch.nn.functional as F
import wandb

import minfnet.util.runtime_util as rtut


##################################
#               model
##################################

class MI_Model(nn.Module):

    def __init__(self, B_N, ctxt_N, encoder_N=128):
        
        super(MI_Model, self).__init__()
        
        # encoder for variable of interest / target (e.g. true energy)
        self.features_a = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, encoder_N), nn.ReLU(),
        )

        # encoder for informing variables (e.g. calo hits)
        self.features_b = nn.Sequential(
            nn.Linear(B_N, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, encoder_N), nn.ReLU(),
        )

        # context conditioning the model (e.g. theta, the detector params)
        self.features_ctxt = nn.Sequential(
            nn.Linear(ctxt_N, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, encoder_N), nn.ReLU(),
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(encoder_N*3, 200),
            nn.ReLU(),
            nn.Linear(200, 1),
            nn.Sigmoid() # todo: try softplus to guarantee positive output (probs), but allow for high correlative values
        )

    def forward(self, a, b, ctxt):
        a = self.features_a(a).view(a.size(0), -1)
        b = self.features_b(b).view(b.size(0), -1)
        ctxt = self.features_ctxt(ctxt).view(ctxt.size(0), -1)
        x = torch.cat((a, b, ctxt), 1) # first dimension is batch-dimension
        return self.fully_connected(x)


def mutual_info(dep_ab, indep_ab, eps=1e-8):

        return dep_ab.mean() - torch.log(indep_ab.exp().mean()+eps) # means over batch



def train(model: MI_Model, dataloader, nb_epochs, optimizer, eps=1e-8):

    model.train()
    
    train_mi = []
    b_c = 0
    
    for e in range(nb_epochs):
        
        acc_mi = 0.0
        
        for b_i, batch in enumerate(dataloader):
            
            batch_a, batch_b, batch_br, theta = [b.to(rtut.device) for b in batch]

            # apply the model: pass a & b and a & b_permuted (conditioned on theta)
            dep_ab = model(batch_a, batch_b, theta)
            indep_ab = model(batch_a, batch_br, theta)

            mi = mutual_info(dep_ab=dep_ab, indep_ab=indep_ab, eps=eps)
            loss = -mi
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        acc_mi += mi.item()
        acc_mi /= len(dataloader)  # mi per batch
        acc_mi /= math.log(2)
        
        wandb.log({"epoch": e, "mi": acc_mi})
        train_mi.append(acc_mi)
        print(f'{e+1} {acc_mi:.04f}')
        sys.stdout.flush()

    return acc_mi


def test(model, dataloader, eps=1e-8):

    model.eval()
    test_acc_mi = 0.0

    for batch in dataloader:

        batch_a, batch_b, batch_br, theta = [b.to(rtut.device) for b in batch]

        dep_ab = model(batch_a, batch_b, theta)
        indep_ab = model(batch_a, batch_br, theta)
        mi = mutual_info(dep_ab=dep_ab, indep_ab=indep_ab, eps=eps)
        test_acc_mi += mi.item()

    test_acc_mi /= len(dataloader)
    test_acc_mi /= math.log(2)

    wandb.log({"test mi": test_acc_mi})

    return test_acc_mi
