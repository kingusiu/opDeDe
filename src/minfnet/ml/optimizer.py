import torch
import numpy as np
from heputl import logging as heplog

logger = heplog.get_logger(__name__)


class Optimizer():

    def __init__(self, surr_dataset, surrogate, theta_nominal, box_covariance=None, step_sz=8e-1, lr=0.01, epoch_n=30):

        self.theta_nominal = theta_nominal
        # pick a random theta from the grid and start with it
        self.theta = surr_dataset.theta[np.random.randint(len(surr_dataset.theta))]
        self.theta.requires_grad_()

        self.surrogate = surrogate
        self.theta_bounds = [np.min(surr_dataset.theta.clone().numpy(), axis=0), \
                              np.max(surr_dataset.theta.clone().numpy(), axis=0)] # [min_t1, min_t2], [max_t1, max_t2]
        self.step_sz = step_sz
        self.optimizer = torch.optim.Adam([self.theta], lr=lr)
        self.epoch_n = epoch_n

        self.box_covariance = np.array(len(theta_nominal)*[1]) if box_covariance is None else box_covariance


    def is_local(self,theta_curr,scaler=2.):
        diff = theta_curr - self.theta_nominal
        return np.dot(diff, np.dot(np.linalg.inv(self.box_covariance), diff)) < scaler
        # return np.all(last_theta >= self.theta_bounds[0]*0.9) and np.all(last_theta <= self.theta_bounds[1]*1.1)

    def optimize(self):
  
        self.surrogate.eval()
        thetas = []
  
        for epoch in range(self.epoch_n):

            thetas.append(self.theta.clone().detach().numpy())

            self.optimizer.zero_grad()
            mi_hat = self.surrogate(self.theta)
            loss = -mi_hat
            
            loss.backward()
            self.optimizer.step()

            theta_curr = self.theta.clone().detach().numpy()
            if ((theta_curr - thetas[-1])**2).sum() < 1e-5:
                break
            if not self.is_local(theta_curr):
                break  
            
            logger.info(f'epoch {epoch}, mi {mi_hat.item():.04f}, theta {theta_curr}')
            
        
        return np.array(thetas)


class Flow_Optimizer(Optimizer):

    def optimize(self, N=100):

        self.surrogate.eval()
        thetas = []

        for epoch in range(self.epoch_n):

            thetas.append(self.theta.clone().detach().numpy())

            self.optimizer.zero_grad
        
            mi_hat, _ = self.surrogate.sample(N,self.theta)

            loss = -mi_hat.mean()

            loss.backward()
            self.optimizer.step()

            theta_curr = self.theta.clone().detach().numpy()
            if ((theta_curr - thetas[-1])**2).sum() < 1e-5:
                break
            if not self.is_local(theta_curr):
                break  
            
            logger.info(f'epoch {epoch}, mi {mi_hat.item():.04f}, theta {theta_curr}')
            
        
        return thetas
    