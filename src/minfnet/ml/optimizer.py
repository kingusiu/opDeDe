import torch
import numpy as np
from heputl import logging as heplog

logger = heplog.get_logger(__name__)


class Optimizer():

    def __init__(self, surr_dataset, surrogate, step_sz=8e-1, lr=0.01, epoch_n=30):
        self.theta = surr_dataset.theta[np.random.randint(len(surr_dataset.theta))]
        self.surrogate = surrogate
        self.theta_bounds = (min(surr_dataset.theta), max(surr_dataset.theta))
        self.step_sz = step_sz
        self.theta.requires_grad_()
        self.optimizer = torch.optim.Adam([self.theta], lr=lr)
        self.epoch_n = epoch_n

    def is_local(self,last_theta):
        return last_theta >= self.theta_bounds[0] and last_theta <= self.theta_bounds[1]

    def optimize(self):
  
        self.surrogate.eval()
        thetas = []
  
        for epoch in range(self.epoch_n):

            thetas.append(self.theta.clone().detach().numpy()[0])

            self.optimizer.zero_grad()
            mi_hat = self.surrogate(self.theta)
            loss = -mi_hat
            
            loss.backward()
            self.optimizer.step()

            if abs(self.theta.item() - thetas[-1]) < 1e-4:
                break
            if not self.is_local(self.theta.item()):
                break  
            
            logger.info(f'epoch {epoch}, mi {mi_hat.item():.04f}, theta {self.theta.item():.04f}')
            
        
        return thetas


